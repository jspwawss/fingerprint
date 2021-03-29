class Logger():
    def __init__(self, path='history.txt', losses=['train_losses','val_losses']):
        self.losses = 
        with open(path, 'w') as log:
            log.write(('epoch,train_losses,val_losses\n'))