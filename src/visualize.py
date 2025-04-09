import matplotlib.pyplot as plt

def visualize_data(train_pth, test_pth, type="train"):
  arr_train, arr_test ="", ""
  with open(train_pth, 'r') as f:
    arr_train = f.read()
  with open(test_pth, 'r') as f:
    arr_test = f.read()
  arr_train, arr_test = arr_train[1:-1], arr_test[1:-1]
  arr_train, arr_test = arr_train.split(", "), arr_test.split(", ")
  data_train = [float(x) for x in arr_train]
  data_test = [float(x) for x in arr_test]
  time = [i for i in range(len(data_train))]
  plt.plot(time, data_train, color='b', label='train')  
  plt.plot(time, data_test, color='r', label='test')  
  plt.xlabel("epoch")
  plt.ylabel("loss")
  plt.title(type)
  plt.legend()
  plt.show()  

visualize_data('result/train_loss.txt', 'result/test_loss.txt', 'train')