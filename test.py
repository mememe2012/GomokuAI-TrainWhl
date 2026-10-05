import NetWork as nt
import GameRule as gamer

def back_func(*args):
    print(args)

if __name__ == '__main__':
    model = nt.InitModel()
    model.CreateModel([128, 64, 16])
    train = nt.TrainModel(model, 2, 5, 0.5, 0.5, winner_func = gamer.checkInput)
    model.model = train.train()
    model.SaveModel("./", "test", {"info" : "test"})