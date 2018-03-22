import pickle
import replay_buffer
def GetBuffer(replay_file):
    with open(replay_file,'rb') as r:
        buffer =  pickle.load(r)
    return buffer
if __name__ == '__main__':
    b = GetBuffer('../data/replay_buffer.pkl')