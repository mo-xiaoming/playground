import os

data_file = os.path.join('data', 'house_tiny.csv')
if not os.path.exists(data_file):
    os.makedirs(os.path.dirname(data_file), exist_ok=True)
    with open(data_file, 'w') as f:
        f.write('NumRooms,Alley,Price\n')  # Column names
        f.write('NA,Pave,127500\n')  # Each row represents a data example
        f.write('2,NA,106000\n')
        f.write('4,NA,178100\n')
        f.write('NA,NA,140000\n')
