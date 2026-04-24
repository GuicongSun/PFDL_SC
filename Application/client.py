#save
import random
from sklearn.model_selection import StratifiedShuffleSplit


# creat_stratifidClients：用 分层 的方式切成若干“份”（保证各份中类别比例接近总体）。
# create_clients + addsamples_client：用 随机打散 + 等块切分 的方式生成若干 shard，然后给每个 shard 追加同一批“公共样本”。

def creat_stratifidClients(feature_list, label_list, num_clients=10, initial='clients'):
    client_names = ['{}_{}'.format(initial, i + 1) for i in range(num_clients)]
    sss = StratifiedShuffleSplit(n_splits=num_clients, test_size=float(1/num_clients), random_state=0)
    shards = []
    for train_index, test_index in sss.split(feature_list, label_list):
        shards.append(list(zip(feature_list[test_index], label_list[test_index])))
    assert (len(shards) == len(client_names))
    return {client_names[i]: shards[i] for i in range(len(client_names))}



def addsamples_client(X_samples, y_samples,shards):
    new_shards = []
    adddata = list(zip(X_samples, y_samples))
    for shard in shards:
        shard.extend(adddata)
        new_shards.append(shard)
    return new_shards



def create_clients(X_samples, y_samples,feature_list, label_list, num_clients=10, initial='clients'):
    ''' return: a dictionary with keys clients' names and value as
                data shards - tuple of images and label lists.
        args:
            image_list: a list of numpy arrays of training images
            label_list:a list of binarized labels for each image
            num_client: number of fedrated members (clients)
            initials: the clients'name prefix, e.g, clients_1

    '''
    # create a list of client names
    client_names = ['{}_{}'.format(initial, i+ 1) for i in range(num_clients)]

    # randomize the data
    data = list(zip(feature_list, label_list))
    #random.seed(0)
    # random.shuffle(data)      # 这个是我注释的，不然极大影响性能----

    # shard data and place at each client
    size = len(data) // num_clients
    shards = [data[i:i + size] for i in range(0, size * num_clients, size)]

    if num_clients!=1:  # 该条件是我加的，为了确保客户端为1时，去掉重复的shample标签样本。
        shards = addsamples_client(X_samples, y_samples, shards)
    # shards = addsamples_client(X_samples, y_samples, shards)


    # number of clients must equal number of shards
    assert (len(shards) == len(client_names))

    return {client_names[i]: shards[i] for i in range(len(client_names))}


if __name__=="__main__":
    import numpy as np
    feature_list = np.array([1,2,3,4,5,6,7,8])
    label_list = np.array([0,1,0,1,1,0,0,0])
    # m = create_clients(feature_list, label_list, num_clients=2, initial='clients')
    m = create_clients(feature_list, label_list,feature_list, label_list, num_clients=2, initial='clients')

    print (m)

    n = creat_stratifidClients(feature_list, label_list, num_clients=2, initial='clients')

    print(n)

    print('d')
