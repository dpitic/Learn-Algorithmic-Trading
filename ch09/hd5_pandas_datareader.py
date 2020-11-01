"""Load financial data and save as HD5 file format."""
import algolib.data as data
import h5py


def main():
    # Load Google financial data into a DataFrame
    goog_data = data.get_google_data(start_date='2001-01-01')
    # Save Google financial data to HD5 file
    hd5_file = 'data/GOOG_data.h5'
    goog_data.to_hdf(path_or_buf=hd5_file, key='goog_data',
                     mode='w', format='table', data_columns=True)
    # Load Google data from HD5 file
    h = h5py.File(hd5_file, 'r')
    print('\nTable information:')
    print(h['goog_data']['table'])
    print('\nTable data:')
    print(h['goog_data']['table'][:])
    print('\nAttributes:')
    for attributes in h['goog_data']['table'].attrs.items():
        print(attributes)


if __name__ == "__main__":
    main()
