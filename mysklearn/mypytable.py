from mysklearn import myutils

# TODO: copy your mypytable.py solution from PA2-PA6 here
import copy
import csv
from statistics import median
from tabulate import tabulate # uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
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

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names) # TODO: fix this

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

        if col_identifier not in self.column_names:
            raise ValueError

        list_cols = []
        for i in range(len(self.data)):
            # This includes NA values
            if include_missing_values:
                list_cols.append(self.data[i][self.column_names.index(col_identifier)])
            # This doesn't include NA values
            else:
                if self.data[i][self.column_names.index(col_identifier)] != "NA":
                    list_cols.append(self.data[i][self.column_names.index(col_identifier)])

        return list_cols # TODO: fix this

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """

        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                try:
                    self.data[i][j] = float(self.data[i][j])
                except:
                    self.data[i][j] = self.data[i][j]

        #pass  TODO: fix this

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """

        for i in reversed(row_indexes_to_drop):
            del self.data[i]

        #pass  TODO: fix this

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

        infile = open(filename)
        reader = csv.reader(infile)

        self.column_names = next(reader)

        for row in reader:
            self.data.append(row)

        infile.close()

        MyPyTable.convert_to_numeric(self)

        # TODO: finish this
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """

        with open(filename, 'w') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(self.column_names)
            writer.writerows(self.data)

        #pass  TODO: fix this

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

        tracked = []
        dup_rows = []

        attr_col = []
        list_col = []

        # Create a list using the selected column
        for i in range(len(self.data)):
            for j in range(len(key_column_names)):
                attr_col.append(self.data[i][self.column_names.index(key_column_names[j])])
            list_col.append(attr_col)
            attr_col = []

        # Check for duplicates
        for i in range(len(list_col)):
            if list_col[i] in tracked:
                dup_rows.append(i)
            else:
                tracked.append(list_col[i])

        return dup_rows # TODO: fix this

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """

        list_index = []
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                if self.data[i][j] == "NA":
                    if i not in list_index:
                        list_index.append(i)
        MyPyTable.drop_rows(self, list_index)

        #pass  TODO: fix this

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """

        sum_list = []

        # Collect data without NA
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                if self.data[i][j] == self.data[i][self.column_names.index(col_name)]:
                    if self.data[i][self.column_names.index(col_name)] != "NA":
                        sum_list.append(self.data[i][j])

        # Averaging data
        data_avg = sum(sum_list) / len(sum_list)

        # Replacing NA with avg
        for i in range(len(self.data)):
            if self.data[i][self.column_names.index(col_name)] == "NA":
                self.data[i][self.column_names.index(col_name)] = data_avg

        #pass  TODO: fix this

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

        better_header = ["attribute", "min", "max", "mid", "avg", "median"]
        better_data = []
        better_stats = []
        
        for k in range(len(col_names)):
            for i in range(len(self.data)):
                for j in range(len(self.data[i])):
                    if self.data[i][j] == self.data[i][self.column_names.index(col_names[k])]:
                        better_data.append(self.data[i][j])
            # append stats
            if(better_data != []):
                min_data = min(better_data)
                max_data = max(better_data)
                mid_data = (min(better_data) + max(better_data))  / 2
                avg_data = sum(better_data) / len(better_data)
                median_data = median(better_data)
                better_stats.append([col_names[k], min_data, max_data, mid_data, avg_data, median_data])
                # empty for next loop
                better_data = []

        return MyPyTable(better_header, better_stats) # TODO: fix this

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """

        for i in other_table.column_names:
            if i not in self.column_names:
                self.column_names.append(i)

        diff_table = []
        for row1 in self.data:
            for row2 in other_table.data:
                for i in key_column_names:
                    if row1[self.column_names.index(i)] == row2[other_table.column_names.index(i)]:
                        key = True
                    else:
                        key = False
                        break
                if key:
                    row3 = set(row2) - set(row1)
                    diff_table.append(row1 + list(row3))

        return MyPyTable(self.column_names, diff_table) # TODO: fix this

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

        col_name_new = []
        for i in self.column_names:
            col_name_new.append(i)
        for i in other_table.column_names:
            if i not in self.column_names:
                col_name_new.append(i)

        new_tab = []
        row_tab = []
        left_tab = []
        right_tab = []
        for i in range(len(self.data)):
            matched = False
            for j in range(len(other_table.data)):
                key = True
                for k in key_column_names:
                    if self.data[i][self.column_names.index(k)] != other_table.data[j][other_table.column_names.index(k)]:
                        key = False
                        break

                if key == True:
                    row_tab.append(self.data[i]) # matched rows
                    row3 = set(other_table.data[j]) - set(self.data[i])
                    new_tab.append(self.data[i] + list(row3))
                    #break
                    matched = True
            
            if matched == False:
                for l in range(len(col_name_new)):
                    if col_name_new[l] in self.column_names:
                        left_tab.append(self.data[i][self.column_names.index(col_name_new[l])])
                    else:
                        left_tab.append("NA")
                new_tab.append(left_tab)
                left_tab = []

        col_attr = []
        col_list = []
        # create rows of elements of only selected columns
        for i in range(len(new_tab)):
            for k in range(len(key_column_names)):
                col_attr.append(new_tab[i][col_name_new.index(key_column_names[k])])
            col_list.append(col_attr)
            col_attr = []

        right_col_attr = []
        right_col_list = []
        # create rows of elements of only selected columns
        for i in range(len(other_table.data)):
            for k in range(len(key_column_names)):
                right_col_attr.append(other_table.data[i][other_table.column_names.index(key_column_names[k])])
            right_col_list.append(right_col_attr)
            right_col_attr = []

        right_tab = []
        for i in range(len(right_col_list)):
            if right_col_list[i] not in col_list:
                for j in range(len(col_name_new)):
                    if col_name_new[j] in other_table.column_names:
                        right_tab.append(other_table.data[i][other_table.column_names.index(col_name_new[j])])
                    else:
                       right_tab.append("NA")
                new_tab.append(right_tab)
                right_tab = []

        joined_result = MyPyTable(col_name_new, new_tab)
        joined_result.pretty_print()
        return joined_result # TODO: fix this