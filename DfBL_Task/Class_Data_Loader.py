class DataLoader:  # class that loads in the data
    def __init__(self, data_set):  # initialise the object with the test and train matrices from the CSV file

        self._data_set = data_set  # data frame
        # the underscore means that the members are protected

        print(self, 'created')  # print statement to show that the object is created within constructor

    def __del__(self):  # destroy object with a print statement
        print(self, 'destroyed')

    #  function that creates a new string column by combining two other columns
    def combine_columns(self, new_column_name, first_column_to_combine, second_column_to_combine):
        self._data_set[new_column_name] = self._data_set[first_column_to_combine].map(str) + " " + \
                                                           self._data_set[second_column_to_combine].map(str)
