import pandas as pd

# Function to convert IPv4 string to integer
def ip_to_int(ip):
    octets = ip.split('.')
    if len(octets) != 4:
        return None
    try:
        int_ip = (int(octets[0]) << 24) + (int(octets[1]) << 16) + (int(octets[2]) << 8) + int(octets[3])
        return int_ip
    except ValueError:
        return None
    
# Function to convert int to bitmask
def int_to_bits(num):
    bits = bin(num)[2:]
    # pad with zeros if needed
    bits = bits.zfill(32)

    # return ".".join(str(bit) for bit in bits)
    return [int(bit) for bit in bits]

# Function to convert IPV4 address to bitmask
def ip_to_bits(ip):
    octets = ip.split('.')
    if len(octets) != 4:
        return None
    int_ip = (int(octets[0]) << 24) + (int(octets[1]) << 16) + (int(octets[2]) << 8) + int(octets[3])
    
    # Convert to bits
    bits = bin(int_ip)[2:]

    # Pad with zeros if needed
    bits = bits.zfill(32)

    # return ".".join(str(bit) for bit in bits)
    return [int(bit) for bit in bits]

# Main
if __name__ == "__main__":
    # Load the dataset
    data = pd.read_csv("data/NF-UQ-NIDS.csv")

    # Convert IP address strings into bitmask
    data['IPV4_SRC_ADDR'] = data['IPV4_SRC_ADDR'].apply(ip_to_bits)
    data['IPV4_DST_ADDR'] = data['IPV4_DST_ADDR'].apply(ip_to_bits)

    # Drop any rows that have NaN values
    old_size = len(data)
    data = data.dropna(how='any')
    print('Entries with NaN dropped: ' + str(old_size-len(data)))

    # Split IP columns into 32 separate columns, one for each bit of address
    data[['sIP31', 'sIP30', 'sIP29', 'sIP28', 'sIP27', 'sIP26', 'sIP25', 'sIP24', 'sIP23', 'sIP22', 'sIP21', 'sIP20', 'sIP19', 'sIP18', 'sIP17', 'sIP16', 'sIP15', 'sIP14', 'sIP13', 'sIP12', 'sIP11', 'sIP10', 'sIP9', 'sIP8', 'sIP7', 'sIP6', 'sIP5', 'sIP4', 'sIP3', 'sIP2', 'sIP1', 'sIP0']] = pd.DataFrame(data['IPV4_SRC_ADDR'].tolist()).astype('bool')
    data[['dIP31', 'dIP30', 'dIP29', 'dIP28', 'dIP27', 'dIP26', 'dIP25', 'dIP24', 'dIP23', 'dIP22', 'dIP21', 'dIP20', 'dIP19', 'dIP18', 'dIP17', 'dIP16', 'dIP15', 'dIP14', 'dIP13', 'dIP12', 'dIP11', 'dIP10', 'dIP9', 'dIP8', 'dIP7', 'dIP6', 'dIP5', 'dIP4', 'dIP3', 'dIP2', 'dIP1', 'dIP0']] = pd.DataFrame(data['IPV4_DST_ADDR'].tolist()).astype('bool')

    # Drop the columns that we converted
    data = data.drop({'IPV4_SRC_ADDR', 'IPV4_DST_ADDR'}, axis=1)

    # Drop any rows that have NaN values, again
    old_size = len(data)
    data = data.dropna(how='any')
    print('Entries with NaN dropped: ' + str(old_size-len(data)))

    # Dataset for attack data
    attacks_data = data.copy()
    attacks_data = attacks_data.groupby('Label')
    attacks_data = attacks_data.get_group(1)
    attacks_data.to_csv('data/cleaned/NF-UQ-NIDS-ATTACKS')
    
    # Dataset for benign data
    benign_data = data.copy()
    benign_data = benign_data.groupby('Label')
    benign_data = benign_data.get_group(0)
    benign_data.to_csv('data/cleaned/NF-UQ-NIDS-BENIGN')

    # Dump the data to a .csv
    data.to_csv('data/cleaned/NF-UQ-NIDS-CLEANED.csv')
    print('Data dumped to CSV')