import json
import os
import csv
import pandas as pd
import hashlib
from ecies.utils import generate_eth_key
from ecies import encrypt, decrypt
privKey = generate_eth_key()
privKeyHe = privKey.to_hex()
pubKeyHex = privKey.public_key.to_hex()

print("Encryption public key:", pubKeyHex)
print("Decryption private key:", privKeyHe)

BLOCKCHAIN_DIR = 'blockchain/'


def get_hash(prev_block):
    with open(BLOCKCHAIN_DIR + prev_block, 'rb') as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def check_integrity():
    files = sorted(os.listdir(BLOCKCHAIN_DIR), key=lambda x: int(x))

    results = []

    for file in files[1:]:
        with open(BLOCKCHAIN_DIR + file) as f:
            block = json.load(f)

        prev_hash = block.get('prev_block').get('hash')
        prev_filename = block.get('prev_block').get('filename')

        actual_hash = get_hash(prev_filename)

        if prev_hash == actual_hash:
            res = 'Ok'
        else:
            res = 'Integrity Breached'

        print(f'Block {prev_filename}: {res}')
        results.append({'block': prev_filename, 'result': res})
    return results


def write_block(policy, policy1, policy2, policy3, policy4, policy5, policy6, policy7, policy8, policy9, policy10, policy11, policy12,
                policy13, policy14):
    blocks_count = len(os.listdir(BLOCKCHAIN_DIR))
    prev_block = str(blocks_count)

    data = {
        "member_name": policy,
        "email": policy1,
        "gender": policy3,
        "location": policy2,
        "employer": policy4,
        "relationship": policy5,
        "patient_name": policy6,
        "patient_suffix": policy7,
        "patient_dob": policy8,
        "cause": policy9,
        "Fee_Charged": policy10,
        "membership_period": policy11,
        "number_of_claims": policy12,
        "number_of_dependants": policy13,
        "label": policy14,
        "prev_block": {
            "hash": get_hash(prev_block),
            "filename": prev_block
        }
    }

    df = [policy, policy1, policy2, policy3, policy4, policy5, policy6, policy7, policy8, policy9, policy10, policy11, policy12,
          policy13, policy14]

    current_block = BLOCKCHAIN_DIR + str(blocks_count + 1)

    with open(current_block, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.write('\n')

    # Writing to CSV excluding hash information
    with open('block.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(df)
    csvFile.close()


def main():
    # write_block(level='Manufacturer', next_level='level1', drug=abc)
    check_integrity()


if __name__ == '__main__':
    main()
