"""mypytable.py code, this is where all the class/method implementation happens"""
import copy
import csv
from tabulate import tabulate

#=====================================================================================
#Jack Brandt
#Course: CPSC 322
#Assignment: PA2
#Date of current version: 9/821/2024
#Did you attempt the bonus? Yes, both of them.
#Brief description of what program does:
#    Implements the methods for MyPyTable class. Including things like loading tables
#  and writing tables to/from csv's, inner/outer joins, calculating stats, and other
#  basic structured data things.
#=====================================================================================

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests

def get_normalizing_function(list_min, list_max):
    '''Create a function for normalizing a list of data
    Args:
        list_min (float): min
        list_max (float): max

    Returns:
        function: A function that takes a value returns it normalized
    '''
    def normalizing_function(value):
        return (value-list_min)/(list_max-list_min)
    return normalizing_function

def int_in_array(ints, i):
    """Looks for i in ints, returns true if found, false otherwise

    Args:
        ints(list of int): list of ints to be searched
        i(int): int to be found

    Returns:
        boolean: Was i found in ints?
    """
    for j in ints:
        if i == j:
            return True
    return False

class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def pretty_string(self):
        '''Like pretty print, except it returns the string instead of printing

        Returns:
            string: tabulate table'''
        return tabulate(self.data, headers=self.column_names)

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)

    def get_index(self, col_identifier):
        """Gets the index for the table based on attribute name

        Args:
            col_identifier(str): the column identifier being looked for
        """

        for i, attribute in enumerate(self.column_names):
            if col_identifier==attribute:
                return i
        raise ValueError('Could not find attribute column')

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        column=[]
        if isinstance(col_identifier,str):
            index = self.get_index(col_identifier)
        else:
            index = col_identifier

        for i,row in enumerate(self.data):
            if not row[index] == 'NA' or include_missing_values:
                column.append(self.data[i][index])
        return column

    def get_row_data_by_column(self, row, column_name):
        """
            Extracts the column data from a row using the column name

            Args:
                row(list(obj)): the row to extract the data from
                column_name(str): the column name

            Returns:
                obj: the data from the row
        """

        return row[self.get_index(column_name)]

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for i,_ in enumerate(self.data):
            for j,_ in enumerate(self.data[i]):
                try:
                    self.data[i][j]=float(self.data[i][j])
                except ValueError:
                    pass #Leave value as is that cannot be converted

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        remaining_data=[]
        for i,_ in enumerate(self.data):
            remove_row=int_in_array(row_indexes_to_drop,i)
            if not remove_row:
                remaining_data.append(self.data[i])
        self.data=remaining_data

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        with open(filename, 'r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            table = []
            for i, row in enumerate(reader):
                if i == 0:
                    self.column_names = row
                else:
                    table.append(row)
            self.data=table
            self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        with open(filename, 'w', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(self.column_names)
            writer.writerows(self.data)

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        column_indexes=[]
        duplicate_indexes=[]
        for key in key_column_names:
            column_indexes.append(self.column_names.index(key))
        for i, row in enumerate(self.data):#Goes through all rows top to bottom
            for j, following_row in enumerate(self.data[i+1:], i+1):#Checks all following rows for duplicates
                is_duplicate = True
                for attribute in column_indexes:#Goes through all attributes
                    #if at least one doesn't match then it's not a duplicate
                    try:
                        if row[attribute] != following_row[attribute]:
                            is_duplicate = False
                    except IndexError:
                        is_duplicate = False
                if (is_duplicate and not j in duplicate_indexes):#Checks to see if duplicate
                    #and that not already in duplicate list
                    duplicate_indexes.append(j)
        return duplicate_indexes

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        full_table=[]
        for row in self.data:
            missing_value=False
            for element in row:
                if element == '' or element == 'NA':
                    #print(element)
                    #print(True)
                    missing_value=True
                    break
            if not missing_value:
                full_table.append(row)
        self.data=full_table

    def remove_rows_where_col_equal_specified(self,column_index, value):
        '''Removes all rows where the value in a specified column matches input value
        Args:
            column_index (int): The index of the column you want to check for a specific value
            value (str): The value you want to check for
        '''
        full_table=[]
        for row in self.data:
            bad_value=False
            if row[column_index] == value:
                    bad_value=True
            if not bad_value:
                full_table.append(row)
        self.data=full_table

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """

        column=self.get_column(col_name)
        column_sum=0
        count=0

        for element in column:#Do average of averagable data
            try:
                if element=='NA':
                    pass
                else:
                    column_sum+=element
                    count+=1
            except TypeError:
                print('couldnt do arithmetic with element')
                print(col_name)
        #print(column_sum)

        for i, element in enumerate(column):#Replace missing values with average
            #print(element)
            if element == 'NA':
                #print(self.get_index(col_name))
                self.data[i][self.get_index(col_name)]=column_sum/count

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Args:
            col_names(list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values should in the columns to compute summary stats
                for should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """

        summary_statistics=MyPyTable(['attribute', 'min','max','mid','avg','median'])
        stats_table=[]
        for attribute in col_names:
            column=self.get_column(attribute, False)
            if not column:#If column empty, return []
                break
            if len(column)%2==1:#if odd take the middle number
                median=sorted(column)[int(len(column)/2)]
            else:#average the two middle numbers if even
                median=(sorted(column)[int(len(column)/2-1)]+sorted(column)[int(len(column)/2)])/2
            stats_table.append([attribute, min(column), max(column), (min(column)+max(column))/2, sum(column)/len(column), median])
        summary_statistics.data=stats_table
        return summary_statistics

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        other_table_key_indexes=[]
        for key in key_column_names:#make keys indexes
            other_table_key_indexes.append(other_table.column_names.index(key))

        joined_headers=self.column_names#make joined header
        for attribute in other_table.column_names:
            if not attribute in key_column_names:
                joined_headers.append(attribute)
        joined_my_py_table=MyPyTable(joined_headers)
        joined_table=[]

        for row in self.data:
            for instance in other_table.data:#Each row in table 1 gets compared to each row in table 2
                has_matching_key=True
                for attribute in key_column_names:#If any of the keys don't match, than its not a match
                    if row[self.column_names.index(attribute)]!=instance[other_table.column_names.index(attribute)]:
                        has_matching_key=False
                if has_matching_key:#Add if matching key
                    joined_row=copy.deepcopy(row)#deep copy bc lists/reference stuff/causes problems otherwise
                    for i, attribute in enumerate(instance):
                        if not i in other_table_key_indexes:
                            joined_row.append(attribute)
                    joined_table.append(copy.deepcopy(joined_row))
        joined_my_py_table.data=joined_table
        return joined_my_py_table

    def check_other_table_for_un_outer_joined(self, other_table, joined_table, key_column_names):
        """Pylint complained about me having too many for/if statements in one function,
            so here's another function.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            joined_table(list of lists): the half-way finished joined_table.
            key_column_names(list of strings):column names to use as key rows

        Returns:
            nothing
        """
        for row in other_table.data:#Checks each row
            exists_in_joined=False
            for instance in joined_table:#Against each row in joined table
                matching_instance=True
                for i, attribute in enumerate(row):
                    if not attribute in instance:
                        matching_instance=False
                if matching_instance:
                    exists_in_joined=True
            if not exists_in_joined:#If it can't find match, carefully add to table
                joined_row=[]
                for attribute in self.column_names:
                    if not attribute in key_column_names:
                        joined_row.append('NA')
                    else:
                        joined_row.append(row[other_table.column_names.index(attribute)])
                for i, attribute in enumerate(row):
                    if not other_table.column_names[i] in key_column_names:
                        joined_row.append(attribute)
                joined_table.append(copy.deepcopy(joined_row))

    def get_other_table_key_indexes(self, key_column_names, other_table):
        """Pylint complained about me having too many for/if statements in one function,
            so here's another function.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of strings):column names to use as key rows

        Returns:
            list of strings: Exactly what the function name describes (:
        """
        other_table_key_indexes=[]#Just turns keys to column indexes list
        for key in key_column_names:
            other_table_key_indexes.append(other_table.column_names.index(key))
        return other_table_key_indexes

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """

        other_table_key_indexes=self.get_other_table_key_indexes(key_column_names, other_table)

        joined_headers=copy.deepcopy(self.column_names)#Make joined header
        for attribute in other_table.column_names:
            if not attribute in key_column_names:
                joined_headers.append(attribute)
        joined_my_py_table=MyPyTable(joined_headers)
        joined_table=[]

        for row in self.data:#basically just inner join, but adds if no matches at end
            row_had_a_match=False
            for instance in other_table.data:#Each row in table 1 gets compared to each row in table 2
                has_matching_key=True
                for attribute in key_column_names:#If any of the keys don't match, than its not a match
                    if row[self.column_names.index(attribute)]!=instance[other_table.column_names.index(attribute)]:
                        has_matching_key=False
                if has_matching_key:#Add if matching key
                    row_had_a_match=True
                    joined_row=copy.deepcopy(row)#deep copy bc lists/reference stuff/causes problems otherwise
                    for i, attribute in enumerate(instance):
                        if not i in other_table_key_indexes:
                            joined_row.append(attribute)
                    joined_table.append(copy.deepcopy(joined_row))
            if not row_had_a_match:#actual outer join stuff, if no match than add with NA to fill
                joined_row=copy.deepcopy(row)
                for  attribute in other_table.column_names:
                    if not attribute in key_column_names:
                        joined_row.append('NA')
                joined_table.append(copy.deepcopy(joined_row))

        #Now just gotta look for not matches of other table
        self.check_other_table_for_un_outer_joined(other_table, joined_table, key_column_names)
        #print(joined_table)
        joined_my_py_table.data=joined_table
        #print(joined_table)
        return joined_my_py_table



    def get_data_subset(self, columns, normalize=True):
        """ Returns a subset of the data with only
        specified columns

        Args:
            columns (list of strings): The names of the
                columns you want.
            normalize (bool): Do you want to normalize
                the data?

        Returns:
            list of lists: Returns a 2d list with shaped:
                number of instances by number of selected
                columns."""
        if normalize is False:
            return [[self.get_column(name)[i] for name in columns]
                    for i in range(self.get_shape()[0])]
        return self.normalize(columns)

    def normalize(self, columns):
        '''Normalizes data, column by column.

        Args:
            data (MyPyTable): The data you want normalized
            columns (list of strings): The names of the columns
                you want normalized

        Returns:
            list of lists: The normalized data
        '''
        normalizing_functions = [get_normalizing_function(
            min(self.get_column(name)),max(self.get_column(name)))
              for name in columns]
        normalized_data = []
        for instance in self.data:
            normalized_instance = []
            function_index = 0
            for i, attribute in enumerate(instance):
                if self.column_names[i] in columns:
                    normalized_instance.append(
                        normalizing_functions[function_index](attribute))
                    function_index+=1
            normalized_data.append(normalized_instance)
        #print(normalized_data)
        return normalized_data

    def fill_from_dict(self, dictionary):
        '''Fills the mypytable attributes from a dictionary
        Args:
            dict (dictionary): Mainly used by classification report for tabulating'''
        self.column_names=['']
        dict_elements=list(dictionary)
        dict_attributes=list(dictionary[dict_elements[0]])
        for attribute in dictionary[dict_elements[0]]:
            self.column_names.append(attribute)
        for element in dict_elements:
            if element in ['accuracy','micro avg']:
                self.data.append(['','','','',''])
            #print([dictionary[element][attribute] for attribute in dict_attributes])
            self.data.append([element]+[round(dictionary[element][attribute],2) if dictionary[element][attribute] != '' else ''
                                        for attribute in dict_attributes])
