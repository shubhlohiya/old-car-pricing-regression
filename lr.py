import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Global variables
phase = "train"  # phase can be set to either "train" or "eval"
km_mean, km_std = 73301.4976 , 62777.3637
km_max, km_min = 2360457, 1000
power_max, power_min = 400.0 , 0.0
torque_max, torque_min = 1450.0 , 48.0
seats_max, seats_min = 14.0 , 4.0
engine_max, engine_min = 3604.0 , 624.0
mileage_max, mileage_min = 42.0 , 0.0
year_max, year_min = 2020 , 1983
mileage_mean, torque_mean, power_mean = 19.375, 175.090, 87.864
seats_mean, engine_mean = 5.456, 1440.076

""" 
You are allowed to change the names of function "arguments" as per your convenience, 
but it should be meaningful.

E.g. y, y_train, y_test, output_var, target, output_label, ... are acceptable
but abc, a, b, etc. are not.

"""

def get_features(file_path):
    # Given a file path , return feature matrix and target labels 
    def get_torque(string):
        if type(string) != str:
            return None
        string = string.lower()
        if r"nm@" in string:
            temp = string.split(r"@")[0]
            if r"nm" in temp:
                return eval(temp.split(r"nm")[0])
            else:
                return eval(temp)
        elif r" nm at" in string:
            return eval(string.split(" nm at")[0])
        elif r"nm at" in string:
            return eval(string.split("nm at")[0])
        elif r"kgm@" in string:
            temp = string.split(r"@")[0]
            if r"kgm" in temp:
                return 10 * eval(temp.split(r"kgm")[0])
            else:
                return 10 * eval(temp)
        elif r" kgm at" in string:
            return 10 * eval(string.split(" kgm at")[0])
        elif r"nm" in string:
            return eval(string.split("nm")[0])
        elif r"@" in string:
            temp = string.split("@")[0]
            if r"(" in temp:
                return eval(temp.split(r"(")[0])
            return eval(temp)
        elif r" /" in string:
            return eval(string.split(" /")[0])
        else:
            return None

    train = pd.read_csv("train.csv", index_col="Index")
    dev = pd.read_csv("dev.csv",index_col="Index")
    test = pd.read_csv("test.csv",index_col="Index")
    train_len, dev_len, test_len = len(train), len(dev), len(test)
    data = pd.concat([train, dev, test], axis=0)


    data["brand"] = data.name.apply(lambda x: x.split()[0])
    data["model"] = data.name.apply(lambda x: x.split()[1])
    data["max_torque"] = data.torque.apply(get_torque)
    data["engine_cc"] = data.engine.apply(lambda x: eval(x.split()[0]) if type(x) == str else None)
    data["mileage_val"] = data.mileage.apply(lambda x: eval(x.split()[0]) if type(x) == str else None)
    fuel_f = pd.get_dummies(data.fuel, prefix='fuel', drop_first=True)
    seller_type_f = pd.get_dummies(data.seller_type, prefix='seller', drop_first=True)
    transmission_f = pd.get_dummies(data.transmission, prefix='transmission', drop_first=True)
    brand_f = pd.get_dummies(data.brand, prefix='brand', drop_first=True)
    model_f = pd.get_dummies(data.model, prefix='model', drop_first=True)
    owner_f = pd.get_dummies(data.owner, prefix='', drop_first=True)
    data = data.drop(columns=["name", "fuel", "seller_type", "transmission", "owner",
                    "torque", "engine", "mileage", "brand", "model"])
    data = pd.concat([data, fuel_f, seller_type_f, transmission_f, owner_f, brand_f, model_f], axis=1)

    train, dev, test = data[:train_len], data[train_len:train_len+dev_len], data[train_len+dev_len:]

    if "train" in file_path:
        data = train
    elif "dev" in file_path:
        data = dev
    elif "test" in file_path:
        data = test

    data.mileage_val = data.mileage_val.fillna(mileage_mean)
    data.max_torque = data.max_torque.fillna(torque_mean)
    data.max_power = data.max_power.fillna(power_mean)
    data.seats = data.seats.fillna(seats_mean)
    data.engine_cc = data.engine_cc.fillna(engine_mean)

    # data.info()

    # Define Scaling Functions
    def normalize(series, mx, mn):
        return (series - mn) / (mx - mn)

    def standardize(series, mean, std):
        return (series - mean) / std
    #
    # # Get scaling parameters
    # sp_mean, sp_std = data.selling_price.mean(), data.selling_price.std()
    # km_mean, km_std = data.km_driven.mean(), data.km_driven.std()
    # power_max, power_min = data.max_power.max(), data.max_power.min()
    # torque_max, torque_min = data.max_torque.max(), data.max_torque.min()
    # seats_max, seats_min = data.seats.max(), data.seats.min()
    # engine_max, engine_min = data.engine_cc.max(), data.engine_cc.min()
    # mileage_max, mileage_min = data.mileage_val.max(), data.mileage_val.min()
    # year_max, year_min = data.year.max(), data.year.min()

    # Perform feature scaling
    data.km_driven = normalize(data.km_driven, km_max, km_min)
    data.max_power = normalize(data.max_power, power_max, power_min)
    data.max_torque = normalize(data.max_torque, torque_max, torque_min)
    data.seats = normalize(data.seats, seats_max, seats_min)
    data.engine_cc = normalize(data.engine_cc, engine_max, engine_min)
    data.mileage_val = normalize(data.mileage_val, mileage_max, mileage_min)
    data.year = normalize(data.year, year_max, year_min)

    phi, y = data.drop(columns=["selling_price"]).to_numpy(), data["selling_price"].to_numpy()
    return phi, y

def get_features_basis(file_path):
    # Given a file path , return feature matrix and target labels 
    phi, y = get_features(file_path)
    phi[:,0:7] = np.exp(phi[:,0:7])
    return phi, y

def compute_RMSE(phi, w , y) :
    # Root Mean Squared Error
    return float(torch.sqrt(torch.sum((1e4 * phi @ w - y) ** 2) / len(y)))

def generate_output(phi_test, w):
    # writes a file (output.csv) containing target variables in required format for Submission.
    preds = 1e4*phi_test@w
    indices = list(range(len(preds)))
    data = np.array([preds, indices]).T
    headers = ["Id", "Expected"]
    pd.DataFrame(data=data, columns=headers).to_csv("output.csv")

def closed_soln(phi, y):
    # Function returns the solution w for Xw=y.
    return np.linalg.pinv(phi).dot(y)

def gradient_descent(phi, y, phi_dev, y_dev, p=5, lr=0.1, cap=5e5, bias=False, verbose=False):
    # Implement gradient_descent using Mean Squared Error Loss
    # You may choose to use the dev set to determine point of convergence

    np.random.seed(123)
    n = phi.shape[0]
    if bias:
        phi = np.append(phi, np.ones((n,1)), axis=1)
        phi_dev = np.append(phi_dev, np.ones((phi_dev.shape[0],1)), axis=1)
    w = np.random.randn(phi.shape[1])
    w = torch.from_numpy(w).cuda(0)
    phi = torch.from_numpy(phi).cuda(0)
    y = torch.from_numpy(y).cuda(0)
    phi_dev = torch.from_numpy(phi_dev).cuda(0)
    y_dev = torch.from_numpy(y_dev).cuda(0)
    rmse_tr = [compute_RMSE(phi, w, y)]
    rmse_dv = [compute_RMSE(phi_dev, w, y_dev)]
    y_prime = y / 1e4

    w_prime = w.clone()
    if verbose:
        print(f"Start..............RMSE on train = {rmse_tr[-1]}, RMSE on dev = {rmse_dv[-1]}")
    i,j,v=0,0,np.inf

    while j<p:
        y_hat = phi @ w
        grad = 2 * phi.t() @ (y_hat - y_prime) / n
        w = w - lr * grad
        i+=1
        if i % 100 == 0:

            v_new = compute_RMSE(phi_dev, w, y_dev)
            if v_new < v:
                j=0
                w_prime = w.clone()
                v = v_new
            else:
                j+=1
            if i % 1000 ==0 and verbose:
                rmse_tr.append(compute_RMSE(phi, w, y))
                rmse_dv.append(v_new)
                print(f"Epoch {i}..............RMSE on train = {rmse_tr[-1]}, RMSE on dev = {rmse_dv[-1]}")
        if i >= cap:
            break

    if i < cap and verbose:
        print("Early Stopping .............. Returning best weights")
    elif verbose:
        print(f"Finished {int(cap)} epochs without convergence...... Returning best weights")
    if verbose:
        plt.plot(rmse_tr)
        plt.plot(rmse_dv)
    return w_prime

def sgd(phi, y, phi_dev, y_dev, p=5, lr=0.03, bs=1, cap=1500, bias=False, verbose=False) :
    # Implement stochastic gradient_descent using Mean Squared Error Loss
    # You may choose to use the dev set to determine point of convergence

    np.random.seed(123)
    m = phi.shape[0]
    n = phi.shape[1]
    if bias:
        phi = np.append(phi, np.ones((m,1)), axis=1)
        phi_dev = np.append(phi_dev, np.ones((phi_dev.shape[0],1)), axis=1)
        w = np.random.randn(n+1)
    else:
        w = np.random.randn(n)
    w = torch.from_numpy(w).cuda(0)
    phi = torch.from_numpy(phi).cuda(0)
    y = torch.from_numpy(y).cuda(0)
    phi_dev = torch.from_numpy(phi_dev).cuda(0)
    y_dev = torch.from_numpy(y_dev).cuda(0)
    rmse_tr = [compute_RMSE(phi, w, y)]
    rmse_dv = [compute_RMSE(phi_dev, w, y_dev)]
    y_prime = y / 1e4

    w_prime = w.clone()
    if verbose:
        print(f"Start..............RMSE on train = {rmse_tr[-1]}, RMSE on dev = {rmse_dv[-1]}")
    i,j,v=0,0,np.inf

    while j<p:
        for k in range(0,m,bs):
            start, end = k*bs, min(m, (k+1)*bs)
            y_hat = phi[start:end] @ w
            grad = phi[start:end].t() @ (y_hat - y_prime[start:end]) / (end-start)
            w = w - lr * grad
        i+=1

        v_new = compute_RMSE(phi_dev, w, y_dev)
        if v_new < v:
            j=0
            w_prime = w.clone()
            v = v_new
        else:
            j+=1
        if verbose:
            rmse_tr.append(compute_RMSE(phi, w, y))
            rmse_dv.append(v_new)
            print(f"Epoch {i}..............RMSE on train = {rmse_tr[-1]}, RMSE on dev = {rmse_dv[-1]}")
        if i >= cap:
            break

    if i < cap and verbose:
        print("Early Stopping .............. Returning best weights")
    elif verbose:
        print(f"Finished {int(cap)} epochs without convergence...... Returning best weights")
    if verbose:
        plt.plot(rmse_tr)
        plt.plot(rmse_dv)
    return w_prime


def pnorm(phi, y, phi_dev, y_dev, p=5, lr=0.03, l=2, lam=1, cap=5e5, bias=False, verbose=False) :
    # Implement gradient_descent with p-norm (here, l-norm) regularisation using Mean Squared Error Loss
    # You may choose to use the dev set to determine point of convergence
    # Constraint on l: l > 1

    np.random.seed(123)
    n = phi.shape[0]
    if bias:
        phi = np.append(phi, np.ones((n,1)), axis=1)
        phi_dev = np.append(phi_dev, np.ones((phi_dev.shape[0],1)), axis=1)
    w = np.random.randn(phi.shape[1])
    w = torch.from_numpy(w).cuda(0)
    phi = torch.from_numpy(phi).cuda(0)
    y = torch.from_numpy(y).cuda(0)
    phi_dev = torch.from_numpy(phi_dev).cuda(0)
    y_dev = torch.from_numpy(y_dev).cuda(0)
    rmse_tr = [compute_RMSE(phi, w, y)]
    rmse_dv = [compute_RMSE(phi_dev, w, y_dev)]
    y_prime = y / 1e4

    w_prime = w.clone()
    if verbose:
        print(f"Start..............RMSE on train = {rmse_tr[-1]}, RMSE on dev = {rmse_dv[-1]}")
    i,j,v=0,0,np.inf

    while j<p:
        y_hat = phi @ w
        r_grad = l*((torch.abs(w)**(l-2))*w)
        if bias:
            r_grad[-1] = 0.
        grad = 2*phi.t() @ (y_hat - y_prime) / n + lam*r_grad
        w = w - lr * grad
        i+=1
        if i % 100 == 0:

            v_new = compute_RMSE(phi_dev, w, y_dev)
            if v_new < v:
                j=0
                w_prime = w.clone()
                v = v_new
            else:
                j+=1
            if i % 1000 ==0 and verbose:
                rmse_tr.append(compute_RMSE(phi, w, y))
                rmse_dv.append(v_new)
                print(f"Epoch {i}..............RMSE on train = {rmse_tr[-1]}, RMSE on dev = {rmse_dv[-1]}")
        if i>=cap:
            break

    if i<cap and verbose:
        print("Early Stopping .............. Returning best weights")
    elif verbose:
        print(f"Finished {int(cap)} epochs without convergence...... Returning best weights")
    if verbose:
        plt.plot(rmse_tr)
        plt.plot(rmse_dv)
    return w_prime
    

def main():
    """ 
    The following steps will be run in sequence by the autograder.
    """
    ######## Task 1 #########
    phase = "train"
    phi, y = get_features('train.csv')
    phase = "eval"
    phi_dev, y_dev = get_features('dev.csv')
    w1 = closed_soln(phi, y)
    w2 = gradient_descent(phi, y, phi_dev, y_dev)
    r1 = compute_RMSE(phi_dev, w1, y_dev)
    r2 = compute_RMSE(phi_dev, w2, y_dev)
    print('1a: ')
    print(abs(r1-r2))
    w3 = sgd(phi, y, phi_dev, y_dev)
    r3 = compute_RMSE(phi_dev, w3, y_dev)
    print('1c: ')
    print(abs(r2-r3))

    ######## Task 2 #########
    w_p2 = pnorm(phi, y, phi_dev, y_dev, 2)  
    w_p4 = pnorm(phi, y, phi_dev, y_dev, 4)  
    r_p2 = compute_RMSE(phi_dev, w_p2, y_dev)
    r_p4 = compute_RMSE(phi_dev, w_p4, y_dev)
    print('2: pnorm2')
    print(r_p2)
    print('2: pnorm4')
    print(r_p4)

    ######## Task 3 #########
    phase = "train"
    phi_basis, y = get_features_basis1('train.csv')
    phase = "eval"
    phi_dev, y_dev = get_features_basis1('dev.csv')
    w_basis = pnorm(phi_basis, y, phi_dev, y_dev, 2)
    rmse_basis = compute_RMSE(phi_dev, w_basis, y_dev)
    print('Task 3: basis')
    print(rmse_basis)

if __name__ == "__main__":
    main()