import sys
import os

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path + '/src/')
sys.path.insert(0, path + '/test/mocks/')
