import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs
from scipy.ndimage.morphology import distance_transform_edt as edt
#计算hd95参数
class HausdorffDistance:
    def hd_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # if not np.any(x):
        #     x[0][0] = 1.0
        # elif not np.any(y):
        #     y[0][0] = 1.0

        indexes = np.nonzero(x)
        distances = edt(np.logical_not(y))

        return np.array(np.percentile(distances[indexes], 95))

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # print(pred.shape,target.shape)
        assert (
            pred.shape[1] == 1 and target.shape[1] == 1
            ), "Only binary channel supported"

        pred = (pred > 0.5).byte()
        target = (target > 0.5).byte()
        if torch.sum(pred) == 0:
            pred[0][0][0][0] = 1
            #print(pred)
            # print(torch.sum(pred))
        # print(pred.shape)
        right_hd = torch.from_numpy(
            self.hd_distance(pred.cpu().numpy(), target.cpu().numpy())
            ).float()

        left_hd = torch.from_numpy(
            self.hd_distance(target.cpu().numpy(), pred.cpu().numpy())
            ).float()

        # print(right_hd, ' ', left_hd)

        return torch.max(right_hd, left_hd)
def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

        out = model(images)
        loss = criterion(out, targets)

        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)
        #iter为批次 这里只有是print_interval的倍数的时候才会输出相应的数据
        if iter % config.print_interval == 0:
            # np.mean计算算数平均值
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step() 
    return step, np.mean(loss_list)


def val_one_epoch(val_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config):
    # switch to evaluate mode
    model.eval()
    #preds out
    #gts ground truth
    preds = []
    gts = []
    loss_list = []
    hd95_list=[]
    mae_list=[]
    with torch.no_grad():
        for data in tqdm(val_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            out2 = np.array(out.cpu())
            msk2=np.array(msk.cpu())
            y_out = np.where(out2 >= config.threshold, 1, 0)
            y_msk = np.where(msk2 >= 0.5, 1, 0)
            y_pre1 = np.array(y_out)  # 预测的二进制标签
            y_true1 = np.array(y_msk)  # 真实的二进制标签
            mae = np.abs(y_pre1 - y_true1)
            mae_list.append(mae)

            loss = criterion(out, msk)

            loss_list.append(loss.item())
            out1=out.float()
            msk1=msk.float()

            hd = HausdorffDistance().compute(out1, msk1)

            hd95_list.append(hd)
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)

    #当轮次到达val_interval的倍数的时候 输出更详细数据 不是则是输出简单信息
    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)
        y_pre1 = np.array(y_pre)  # 预测的二进制标签
        y_true1 = np.array(y_true)  # 真实的二进制标签
        mae = np.abs(y_pre1 - y_true1).mean()
        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
        Precision = float(TP) / float(TP + FP) if float(TP + FP) != 0 else 0
        Recall = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0

        log_info = (f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, precision: {Precision},recall: {Recall},hd95: {np.mean(hd95_list)}accuracy: {accuracy}'f'specificity: {specificity}, sensitivity: {sensitivity},MAE:{np.mean(mae_list)}, confusion_matrix: {confusion}')
        print(log_info)
        logger.info(log_info)

    else:
        #np.mean计算算术平均值
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)
    
    return np.mean(loss_list)


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    hd95_list = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            out1 = out.float()
            msk1 = msk.float()

            hd = HausdorffDistance().compute(out1, msk1)
            hd95_list.append(hd)
            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)
            if i % config.save_interval == 0:
                save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)
        y_pre1 = np.array(y_pre)  # 预测的二进制标签
        y_true1 = np.array(y_true)  # 真实的二进制标签
        mae = np.abs(y_pre1 - y_true1).mean()
        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
        Precision = float(TP) / float(TP + FP) if float(TP + FP) != 0 else 0
        Recall = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc},hd95: {np.mean(hd95_list)} ,accuracy: {accuracy}, specificity: {specificity}, sensitivity: {sensitivity},MAE:{mae},precision: {Precision},recall: {Recall},confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)