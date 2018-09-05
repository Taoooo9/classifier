import torch.optim
import time
import math
import torch.nn.functional as F
from dataloader.dataloading import *

def train(model, train_data, dev_data, test_data, src_vocab, target_vocab, args):
    model.train()
    if args.learning_algorithm == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)

    elif args.learning_algorithm == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr = args.lr)

    elif args.learning_algorithm == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    elif args.learning_algorithm == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr == args.lr)
    else:
        raise RuntimeError("Invalid optim method: " + args.learning_algorithm)

    global_step = 0
    best_acc = 0
    print('\nstart training...')

    for once in range(args.epochs):
        once_start_time = time.time()
        print('epoch:', once)
        batch_num = int(math.ceil(len(train_data) / float(args.batch_size)))

        batch_iter = 0
        for batch in create_batch(train_data, args.batch_size, shuffle = True):
            start_time = time.time()
            feature, src_target, feature_length = pair_data_variable(batch, src_vocab, target_vocab, args)

            optimizer.zero_grad()
            logit = model(feature)
            l = F.cross_entropy(logit, src_target)
            l.backward()
            optimizer.step()

            correct = (torch.max(logit, 1)[1].view(src_target.size()).data == src_target.data).sum()
            accuracy = 100.0 * correct / len(batch)

            during_time = float(time.time() - start_time)
            print("Step:{}, Epoch:{}, batch:{}, accuracy:({:.4f}%)({}/{}), time:{:.2f}, loss:{:.6f}"
                  .format(global_step, once, batch_iter, accuracy, correct, len(batch), during_time, l.item()))

            batch_iter += 1
            global_step += 1

            if batch_iter % args.test_interval == 0 or batch_iter == batch_num:
                dev_acc = evaluate(model, dev_data, global_step, src_vocab, target_vocab, args)
                test_acc = evaluate(model, test_data, global_step, src_vocab, target_vocab, args)

                if dev_acc > best_acc:
                    print("Exceed best acc: history = %.2f, current = %.2f" % (best_acc, dev_acc))
                    best_acc = dev_acc
                    if args.save_after > 0 and once > args.save_after:
                        torch.save(model.state_dict(), args.save_model_path + '.' + str(global_step))

        during_time = float(time.time() - once_start_time)
        print('one iter using time: time:{:.2f}'.format(during_time))


def evaluate(model, data, step, src_vocab, target_vocab, args):
    model.eval()
    start_time = time.time()
    correct, size = 0, 0

    for batch in create_batch(data, args.batch_size):
        feature, target, feature_len = pair_data_variable(batch, src_vocab, target_vocab, args)
        logit = model(feature)
        #correct += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

        pred = logit.data.max(1, keepdim = True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        size += len(batch)
    accuracy = 100.0 * correct / size
    during_time = time.time() - start_time
    print("\nevaluate result: ")
    print("Step:{}, accuracy:({:.4f}%)({}/{}), time:{:.2f}"
          .format(step, accuracy, correct, size, during_time))
    model.train()
    return accuracy







