class DataLoader:  # class that loads in the data
    def __init__(self, data_set):  # initialise the object with the test and train matrices from the CSV file

        self._data_set = data_set  # dataframe
        self._target = 0  # target attribute
        self._id = 0  # id of each column
        # the underscore means that the members are protected

        print(self, 'created')  # print statement to show that the object is created within constructor

    def __del__(self):  # destroy object with a print statement
        print(self, 'destroyed')
