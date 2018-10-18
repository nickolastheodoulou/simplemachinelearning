from sklearn.utils import shuffle


class DataLoader:  # class that stores the data set as an object. The purpose of this class is to
    def __init__(self, data_set):  # initialise the object with the data_set

        self._data_set = data_set  # data set
        # the underscore means that the members are protected
        print(self, 'created')

    #  method that shuffles the data set
    def shuffle_data_set(self):
        # shuffle using sklearn.utils, seed set to 0 to get the same shuffle each time to test model
        self._data_set = shuffle(self._data_set, random_state=1)

    def __del__(self):
        print(self, 'destroyed')  # print statement when the destructor is called
