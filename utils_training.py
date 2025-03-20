import torch.nn.functional as F
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import numpy.linalg as la
from scipy.ndimage import zoom
from tqdm import tqdm

class Torch_Dataset(Dataset):
    def __init__(self, data, labels, dev) :
        device = torch.device(dev if torch.cuda.is_available() else "cpu")
        self.iscomplex = (True if (np.max(data.imag)>1e-15) else False)
        self.labels =  torch.tensor(labels, dtype = int, device = device)
    
        if self.iscomplex == False:
            self.data = torch.tensor(data.real, dtype = torch.float32, device = device)
        else:
            self.data = torch.tensor(np.array([np.concatenate([im.real.flatten(), im.imag.flatten()]) for im in data]), dtype = torch.float32, device = device)
                
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class LR_classifier(nn.Module):
    def __init__(self, input_size : int, n_outs : int, iscomplex: bool):
        super(LR_classifier, self).__init__()
        self.input_size = input_size
        self.n_outputs = n_outs
        self.iscomplex = iscomplex
        #
        if iscomplex:
            self.classifier = nn.Linear(2 * self.input_size, self.n_outputs)
        else:
            self.classifier = nn.Linear(self.input_size, self.n_outputs)
    def forward(self, X):
        # Y = torch.abs(self.fft2_centered(X))
#         plt.figure()
#         plt.imshow(Y[0].detach().cpu(), cmap = 'gray')
#         plt.colorbar()
#         plt.show()
        if self.iscomplex:
            Y = self.classifier(X.reshape(X.shape[0], 2 * self.input_size))
        else:
            Y = self.classifier(X.reshape(X.shape[0], self.input_size))
        return Y


def get_accuracy(logit, target):
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects
    return accuracy.item()


def train_test_generator(data_images, data_labels, NTrain=None, NTest=None, rand=True, rangeTest=None, returnIndices=False):
    N_total_samp = len(data_images)
    N_train_samp = (NTrain if (NTrain and NTrain + NTest <= N_total_samp) else int(0.8 * N_total_samp))
    N_test_samp = (NTest if (NTest and NTrain + NTest <= N_total_samp) else int(0.2 * N_total_samp))

    if rand:
        indices = np.random.permutation(N_total_samp)
    else:
        indices = np.arange(N_total_samp)

    if rangeTest is None:
        test_indices = indices[:N_test_samp]
    else:
        i = 0
        test_indices = []
        while len(test_indices) < N_test_samp:
            if indices[i] in rangeTest:
                test_indices.append(indices[i])
            i = i + 1
    test_indices = np.array(test_indices)
    train_indices = np.setdiff1d(indices[:(N_train_samp + N_test_samp)], test_indices)
    
    train_images = data_images[train_indices]
    train_labels = data_labels[train_indices]
    test_images = data_images[test_indices]
    test_labels = data_labels[test_indices]

    if returnIndices:
        return (train_images, train_labels, test_images, test_labels, train_indices, test_indices)
    else:
        return (train_images, train_labels, test_images, test_labels)


def downsample_data(data, L=None, dim="2D"):
    """
    Input:
    data: N_total_samp * L_max * L_max or N_total_samp * L_max, np.array, with image size (for 2D) or sequence length (for 1D)
    L: Target downsampled size (either L*L for 2D or L for 1D)
    dim: "2D" for 2D images, "1D" for 1D sequences
    Output:
    Downsampled data, either L*L pixels for 2D or L samples for 1D
    """
    if dim == "2D":
        L_max = np.shape(data)[1]  # Assume data is N_total_samp * L_max * L_max
        L_samp = (L if (L and L <= L_max) else L_max)
        return np.array([zoom(im, L_samp / L_max, order=1) for im in data]).reshape(-1, int(L_samp ** 2))
    
    elif dim == "1D":
        L_max = np.shape(data)[1]  # Assume data is N_total_samp * L_max for 1D
        L_samp = (L if (L and L <= L_max) else L_max)
        return np.array([zoom(seq, L_samp / L_max, order=1) for seq in data]).reshape(-1, int(L_samp))

    else:
        raise ValueError("dim should be '1D' or '2D'")



def centered_image_data(data, L=None):
    """
    Input:
    data: N_total_samp * L_max * L_max, np.array, with image size
    Output:
    centered L*L pixels in each image
    """
    L_max = np.shape(data)[1]
    L_samp = (L if (L and L<=L_max) else L_max)
    
    return data[:,
                L_max // 2 - L_samp // 2:L_max // 2 - L_samp // 2 + L_samp, 
                L_max // 2 - L_samp // 2:L_max // 2 - L_samp // 2 + L_samp].reshape(-1, int(L_samp ** 2))


def LogisticTrain(data_set, data_labels, dev="cuda:1", init_lr=4e-2, Epochs=600, manual_seed=None, K=None, NTrain=None, NTest=None, verbose=False, rand=True, justTrain=False, rangeTest=None, runningAccuracy=False, batch_size=100):
    '''
    manual_seed: random seed, default none
    K: number of features used in training
    NTrain, NTest: training, test set size
    rand: if True, then randomly divide data_set and data_labels into two sets for training and testing; else, the first NTrain is training and the following NTest is testing
    verbose: output running accuracy every 10% progress
    justTrain: no output at all (if False, then output progress in terms of epochs run)
    n_outs: by default 10, should be written as number of unique labels, but now as manual input argument
    '''
    device = torch.device(dev if torch.cuda.is_available() else "cpu")
    iscomplex = (True if (np.max(data_set.imag)>1e-15) else False)

    N_total_samp = len(data_set)
    N_train_samp = (NTrain if (NTrain and NTrain + NTest <= N_total_samp) else int(0.8 * N_total_samp))
    N_test_samp = (NTest if (NTest and NTrain + NTest <= N_total_samp) else int(0.2 * N_total_samp))
    K_samp = (len(data_set[0]) if (K is None or K>len(data_set[0])) else K)
    # print(K_samp)

    train_images, train_labels, test_images, test_labels = train_test_generator(data_set, data_labels, NTrain=NTrain, NTest=NTest, rand=rand, rangeTest=rangeTest)
    n_outs = len(np.unique(train_labels))

    # print(test_labels)
    # for data in [train_images, train_labels, test_images, test_labels]:
    #     print(data.shape)

    train_data = Torch_Dataset(train_images[:, :K_samp], train_labels, dev=device)
    test_data = Torch_Dataset(test_images[:, :K_samp], test_labels, dev=device)
    train_dataset = DataLoader(train_data, batch_size = batch_size, shuffle = True)
    test_dataset = DataLoader(test_data, batch_size = batch_size, shuffle = False)

    model_lin = LR_classifier(K_samp, n_outs = n_outs, iscomplex = iscomplex).to(device)

    #Load Data
    if manual_seed:
        torch.manual_seed(manual_seed)

    learn_rate = init_lr
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model_lin.parameters(), lr=learn_rate, betas = (0.999, 0.999), weight_decay = 0.0)
    #optimizer = torch.optim.SGD(model_lin.parameters(), lr=learn_rate, weight_decay = 0.0)

    # if (device != "cpu") :
    #     with torch.cuda.device(dev):
    #         torch.cuda.empty_cache()
            
    loss_train = []
    loss_test = []
    acc_train = []
    acc_test = []
    photon_avg = []

    for epoch in (range(Epochs) if (verbose or justTrain) else tqdm(range(Epochs))):# loop over the dataset multiple times
        optimizer.lr = learn_rate/(epoch + 1)**0.5
        if epoch == 60 :
            learn_rate = init_lr / 4
        if epoch == 120 :
            learn_rate = init_lr / 25
        if epoch == 300 :
            learn_rate = init_lr / 50
        if epoch == 500 :
            learn_rate = init_lr / 100
        if epoch == 1000 :
            learn_rate = init_lr / 200
        train_running_loss = 0.0
        train_acc = 0.0
        test_acc = 0.0
        model_lin.train()
        for i, data in enumerate(train_dataset):
            optimizer.zero_grad()
            inputs, labels = data
            outputs = model_lin(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if runningAccuracy or epoch == Epochs-1:
                train_running_loss += loss.detach().item()
                train_acc += get_accuracy(outputs, labels)
        model_lin.eval()
        if runningAccuracy:
            acc_train.append(train_acc/N_train_samp)
            loss_train.append(train_running_loss/(N_train_samp))
            if verbose and (epoch%(Epochs/10) == 0):
                print('Progress:  %d | Loss: %.4f | Accuracy: %.2f'\
                    %(100*epoch/Epochs, train_running_loss/(N_train_samp), \
                        train_acc/N_train_samp))
        t_loss = 0.
        t_acc = 0.
        if runningAccuracy or epoch == Epochs-1:
            with torch.no_grad():
                for p, tdata in enumerate(test_dataset):
                    t_inputs, t_labels = tdata
                    t_outputs = model_lin(t_inputs)
                    t_loss += criterion(t_outputs, t_labels).detach().item()
                    test_acc += get_accuracy(t_outputs, t_labels)
                acc_test.append(test_acc/N_test_samp)
                loss_test.append(t_loss/(N_test_samp))
        if verbose or epoch == Epochs-1:
            if verbose and (epoch%(Epochs/10) == 0) :
                print('Test Loss: %.6f | Test Accuracy: %.4f' 
                %(t_loss/(N_test_samp), test_acc/N_test_samp))
    
    if runningAccuracy:
        return (train_acc/N_train_samp, test_acc/N_test_samp), (np.array(acc_train), np.array(acc_test))
    else:
        return (train_acc/N_train_samp, test_acc/N_test_samp)


def eigentask_solver(data, V=None):
    '''
    data: of shape N * (some shape), where N is the number of samples, usually (N * K)
    V: covariance matrix, of shape K * K

    Return:
    data_orig_basis: basically the original data, but reshaped to a 2D numpy array
    data_eigen_basis: data in eigentask basis of shape (N * K), ordered by NSR values
    nsr_nocorrection: PLEASE IGNORE THIS, TO BE MODIFIED
    r_train.T: eigentask masks, each row is a mask for one eigentask
    '''

    #Assumed Poisson distribution
    XMat = data.reshape(len(data), -1).T
    G = XMat @ XMat.T / XMat.shape[1]
    VPoisson = np.diag(np.mean(XMat, axis=1))

    D = (V if V is not None else VPoisson)
    RandomWalk = la.pinv(D) @ G
    s, r = la.eig(RandomWalk)
    idx = s.argsort()[::-1]
    s_train = np.real(s[idx])
    r_train = np.real(r[:,idx])

    nsr_nocorrection = s_train
    data_orig_basis = XMat.T
    data_eigen_basis = XMat.T @ r_train
    # data_eigen_basis = np.array([data_eigen_basis[i] / np.sum(data_eigen_basis[i]) * np.sum(data_orig_basis[i]) for i in range(len(data_orig_basis))])

    return data_orig_basis, data_eigen_basis, nsr_nocorrection, r_train.T

def pca_solver(data, zero_center=True):
    if zero_center == True:
        pca = PCA()
        data_pca = pca.fit_transform(data.reshape(len(data), -1))
        return data_pca, pca
    else:
        data = data.reshape(len(data), -1)
        M = data.T @ data
        eigs, pc = np.linalg.eig(M)
        idx = np.argsort(eigs)[::-1]
        pc = (pc.T[idx]).T
        return data @ pc, pc

def LinearRegression(data_images, data_labels, K=None, NTrain=None, NTest=None, rand=True, rangeTest=None, bias=True, returnW=False):
    n_outs = len(np.unique(data_labels))
    if isinstance(data_labels[0], np.int64) or isinstance(data_labels[0], int):
        labels_onehot = np.array([[1 if i == label else 0 for i in range(n_outs)] for label in data_labels])
    else:
        labels_onehot = data_labels
    X = (np.array(data_images).reshape(len(data_images), -1))
    K_samp = X.shape[1] if (K is None) else K
    X = X[:, :K_samp]
    if bias:
        X = np.concatenate((X.T, np.ones((1, len(data_images))))).T
    Y = np.array(labels_onehot)
    XTrain, YTrain, XTest, YTest = train_test_generator(X, Y, NTrain=NTrain, NTest=NTest, rand=rand, rangeTest=rangeTest)
    W = np.linalg.pinv(XTrain.T @ XTrain) @ XTrain.T @ YTrain
    acc_train = np.sum(np.argmax(XTrain @ W, axis=1) == np.argmax(YTrain, axis=1)) / NTrain * 100
    acc_test = np.sum(np.argmax(XTest @ W, axis=1) == np.argmax(YTest, axis=1)) / NTest * 100
    if returnW:
        return acc_train, acc_test, W
    else:
        return acc_train, acc_test

def low_pass_2D(FFTdata, L=None, type="real"):
    """
    Input:
    FFFdata: N_total_samp * L_max * L_max, np.array, with image size, should be shifted FFT data, real
    Output:
    real and imag parts of centered
    """
    L_max = np.shape(FFTdata)[1]
    L_samp = (L if (L and L<=L_max) else L_max)

    if type == "real":
        freqs_real = FFTdata[:, 
                        L_max // 2 - L_samp // 2:L_max // 2 - L_samp // 2 + L_samp, 
                        L_max // 2:L_max // 2 + (L_samp + 1) // 2].real
        
        freqs_imag = FFTdata[:, 
                        L_max // 2 - L_samp // 2:L_max // 2 - L_samp // 2 + L_samp, 
                        L_max // 2:L_max // 2 + L_samp // 2].imag
        
        return np.concatenate((freqs_real.reshape(len(FFTdata), -1), freqs_imag.reshape(len(FFTdata), -1)), axis=1)
    

def low_pass_1D(FFTdata, L=None, type="real"):
    """
    Input:
    FFTdata: N_total_samp * L_max, np.array, with sequence length, should be shifted FFT data, real
    L: Length of the low-pass filter (number of frequencies to keep)
    type: "real" for real part extraction and low-pass filtering
    Output:
    Real and imaginary parts of the centered low-pass filtered data
    """
    L_max = np.shape(FFTdata)[1]
    L_samp = (L if (L and L <= L_max) else L_max)

    if type == "real":
        # Extract the real and imaginary parts for low-pass filtering
        freqs_real = FFTdata[:, 
                             L_max // 2:L_max // 2 + L_samp // 2 + 1].real
                             
        freqs_imag = FFTdata[:, 
                             L_max // 2:L_max // 2 - L_samp // 2 + L_samp].imag
        
        return np.concatenate((freqs_real.reshape(len(FFTdata), -1), 
                               freqs_imag.reshape(len(FFTdata), -1)), axis=1)


class DNN(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        Nunits=None,
        batchnorm=False,
        nlaf="relu",
        final_sigmoid=False,
        **kwargs
    ):
        """
        Defines configurable deep neural network with fully connected layers and a choice of
        nonlinear activation functions.

        Args:
            input_dim (int): dimension of input layer
            output_dim (int): dimension of output layer
            Nunits (list of int): dimensions of hidden layers
            batchnorm (bool): determines whether to use batchnorm between each hidden layer.
                The order in which batchnorm is applied is:
                fully connected layer - batchnorm - nonlinear activation function
            nlaf (string): determines the nonlinear activation function. Choices:
                'relu', 'tanh', 'sigmoid'
        """
        super(DNN, self).__init__()

        if Nunits == None:
            Nunits = [100, 100]
        self.batchnorm = batchnorm
        # set nonlinear activation function
        if nlaf == "relu":
            self.nlaf = torch.relu
        elif nlaf == "tanh":
            self.nlaf = torch.tanh
        elif nlaf == "sigmoid":
            self.nlaf = torch.sigmoid

        Nunits.insert(0, input_dim)

        self.layers = nn.ModuleList([])
        for i in range(len(Nunits) - 1):
            self.layers.append(nn.Linear(Nunits[i], Nunits[i + 1]))
        self.outputlayer = nn.Linear(Nunits[-1], output_dim)

        if batchnorm:
            self.batchnorms = nn.ModuleList([])
            for i in range(len(Nunits) - 1):
                self.batchnorms.append(nn.BatchNorm1d(Nunits[i + 1]))

    def forward(self, x):
        """
        Performs the forward pass through the network.

        Args:
            x (float tensor): inputs of dimension [batch_size, input_dim]
        """

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.batchnorm:
                x = self.batchnorms[i](x)
            x = self.nlaf(x)

        return self.outputlayer(x)


def DNNTrain(
    data_set,
    data_labels,
    modelargs,
    dev="cuda:1",
    init_lr=4e-2,
    Epochs=600,
    manual_seed=None,
    K=None,
    NTrain=None,
    NTest=None,
    verbose=False,
    rand=True,
    justTrain=False,
    rangeTest=None,
    runningAccuracy=False,
    batch_size=100,
    Model=DNN,
):
    """
    manual_seed: random seed, default none
    K: number of features used in training
    NTrain, NTest: training, test set size
    rand: if True, then randomly divide data_set and data_labels into two sets for training and testing; else, the first NTrain is training and the following NTest is testing
    verbose: output running accuracy every 10% progress
    justTrain: no output at all (if False, then output progress in terms of epochs run)
    n_outs: by default 10, should be written as number of unique labels, but now as manual input argument
    """
    device = torch.device(dev if torch.cuda.is_available() else "cpu")
    iscomplex = True if (np.max(data_set.imag) > 1e-15) else False

    N_total_samp = len(data_set)
    N_train_samp = (
        NTrain
        if (NTrain and NTrain + NTest <= N_total_samp)
        else int(0.8 * N_total_samp)
    )
    N_test_samp = (
        NTest if (NTest and NTrain + NTest <= N_total_samp) else int(0.2 * N_total_samp)
    )
    K_samp = len(data_set[0]) if (K is None or K > len(data_set[0])) else K
    # print(K_samp)

    train_images, train_labels, test_images, test_labels = train_test_generator(
        data_set,
        data_labels,
        NTrain=NTrain,
        NTest=NTest,
        rand=rand,
        rangeTest=rangeTest,
    )
    n_outs = len(np.unique(train_labels))

    # print(test_labels)
    # for data in [train_images, train_labels, test_images, test_labels]:
    #     print(data.shape)

    train_data = Torch_Dataset(train_images[:, :K_samp], train_labels, dev=device)
    test_data = Torch_Dataset(test_images[:, :K_samp], test_labels, dev=device)
    train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataset = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # model_lin = LR_classifier(K_samp, n_outs=n_outs, iscomplex=iscomplex).to(device)
    model = Model(K_samp * 2 if iscomplex else K_samp, n_outs, **modelargs).to(device)

    # Load Data
    if manual_seed:
        torch.manual_seed(manual_seed)

    learn_rate = init_lr
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learn_rate, betas=(0.999, 0.999), weight_decay=0.0
    )
    # optimizer = torch.optim.SGD(model_lin.parameters(), lr=learn_rate, weight_decay = 0.0)

    # if (device != "cpu") :
    #     with torch.cuda.device(dev):
    #         torch.cuda.empty_cache()

    loss_train = []
    loss_test = []
    acc_train = []
    acc_test = []
    photon_avg = []

    for epoch in (
        range(Epochs) if (verbose or justTrain) else tqdm(range(Epochs))
    ):  # loop over the dataset multiple times
        optimizer.lr = learn_rate / (epoch + 1) ** 0.5
        if epoch == 60:
            learn_rate = init_lr / 4
        if epoch == 120:
            learn_rate = init_lr / 25
        if epoch == 300:
            learn_rate = init_lr / 50
        if epoch == 500:
            learn_rate = init_lr / 100
        if epoch == 1000:
            learn_rate = init_lr / 200
        train_running_loss = 0.0
        train_acc = 0.0
        test_acc = 0.0
        model.train()
        for i, data in enumerate(train_dataset):
            optimizer.zero_grad()
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if runningAccuracy or epoch == Epochs - 1:
                train_running_loss += loss.detach().item()
                train_acc += get_accuracy(outputs, labels)
        model.eval()
        if runningAccuracy:
            acc_train.append(train_acc / N_train_samp)
            loss_train.append(train_running_loss / (N_train_samp))
            if verbose and (epoch % (Epochs / 10) == 0):
                print(
                    "Progress:  %d | Loss: %.4f | Accuracy: %.2f"
                    % (
                        100 * epoch / Epochs,
                        train_running_loss / (N_train_samp),
                        train_acc / N_train_samp,
                    )
                )
        t_loss = 0.0
        t_acc = 0.0
        if runningAccuracy or epoch == Epochs - 1:
            with torch.no_grad():
                for p, tdata in enumerate(test_dataset):
                    t_inputs, t_labels = tdata
                    t_outputs = model(t_inputs)
                    t_loss += criterion(t_outputs, t_labels).detach().item()
                    test_acc += get_accuracy(t_outputs, t_labels)
                acc_test.append(test_acc / N_test_samp)
                loss_test.append(t_loss / (N_test_samp))
        if verbose or epoch == Epochs - 1:
            if verbose and (epoch % (Epochs / 10) == 0):
                print(
                    "Test Loss: %.6f | Test Accuracy: %.4f"
                    % (t_loss / (N_test_samp), test_acc / N_test_samp)
                )

    if runningAccuracy:
        return (train_acc / N_train_samp, test_acc / N_test_samp), (
            np.array(acc_train),
            np.array(acc_test),
        )
    else:
        return (train_acc / N_train_samp, test_acc / N_test_samp)