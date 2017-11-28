'''
Created on 26-Mar-2017

@author: aniron
'''

class TestClass(object):
    '''
    This is a test documentation
    '''

    def __init__(self, *args, **kwargs):
        '''
        Constructor
        '''
        pass
    
    def main_test(self):
        return 'Hello world!'

if __name__ == '__main__':
    t = TestClass()
    print t.main_test()
