import PLA

data_train_path = 'HW1/dataset/data_train.txt'
data_test_path = 'HW1/dataset/data_test.txt'

x_train, y_train = PLA.preprocess(data_train_path)
x_test, y_test = PLA.preprocess(data_test_path)

PLA.f2(PLA.Pocket_PLA, x_train, y_train, x_test, y_test, 2000, max_step=50)

PLA.f2(PLA.PLA, x_train, y_train, x_test, y_test, 2000, 1, max_step=50)

PLA.f2(PLA.Pocket_PLA, x_train, y_train, x_test, y_test, 2000, 1, max_step=50)
