import hashlib
import time

# 블록 클래스
class Block:
    def __init__(self, index, previousHash, data):
        self.index = index
        self.previousHash = previousHash
        self.data = data
        self.timestamp = int(time.time())
        self.hash = self.calculateHash()

    def calculateHash(self):
        data_string = str(self.index) + self.previousHash + str(self.timestamp) + self.data
        sha = hashlib.sha256(data_string.encode()).hexdigest()
        return sha

# 블록체인 클래스
class Blockchain:
    def __init__(self):
        self.chain = [self.createGenesisBlock()]

    def createGenesisBlock(self):
        return Block(0, "0", "Genesis Block")

    def getLatestBlock(self):
        return self.chain[-1]

    def addBlock(self, newBlock):
        newBlock.previousHash = self.getLatestBlock().hash
        newBlock.hash = newBlock.calculateHash()
        self.chain.append(newBlock)

if __name__ == "__main__":
    myBlockchain = Blockchain()

    myBlockchain.addBlock(Block(1, "", "Data 1"))
    myBlockchain.addBlock(Block(2, "", "Data 2"))

    for block in myBlockchain.chain:
        print("Block #", block.index)
        print("Timestamp: ", time.ctime(block.timestamp))
        print("Data: ", block.data)
        print("Previous Hash: ", block.previousHash)
        print("Hash: ", block.hash)
        print("----------------------------------")
