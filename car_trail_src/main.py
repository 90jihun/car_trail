# -*- coding: utf-8 -*-
# +
import torch
from torch import nn
import numpy as np
import torchvision
import tqdm
from torch.utils.data import Dataset, DataLoader
import cv2
import os
from glob import glob
import matplotlib.pyplot as plt
# %matplotlib inline
import utm
from model import CNN_LSTM
from model_left import CNN_LSTM_LEFT
from model_right import CNN_LSTM_RIGHT

device = 'cuda:0'

class LoadDataset(Dataset):
    def __init__(self, method=None):
        # 데이터 위치에 따라 root를 변경
        self.root = './RF_0/'
        self.img_seq = []
        self.pos_seq = []
        self.gt_seq = []
        self.states = np.array([])
        
        if method == 'train':       
            self.root = self.root + 'train/'
                
        elif method == 'test': 
            self.root = self.root + 'test/'
                
        # input data path 정렬
        num_dir = glob(self.root + '*')
        sort_dir = sorted(num_dir, key = lambda j : int(os.path.splitext(os.path.basename(j))[0]))
        self.img_path=[]
        
        for i in tqdm.tqdm(range(len(sort_dir))):
            # 이미지 path를 숫자순으로 정렬
            path = glob(sort_dir[i] + '/image1/*.png')
            sort_path = sorted(path,key = lambda j : int(os.path.splitext(os.path.basename(j))[0]))
            self.img_path.append(sort_path)
            
            # 폴더 내 이미지의 수가 170장 미만이면 제외
            if len(self.img_path[i]) < 170:
                continue
                
            self.utm_pos = []
            img_resize = []
                
            # 이미지를 불러온 뒤 resize하여 리스트에 저장
            for j in range(len(self.img_path[i])):
                if len(self.img_path[i]) - j > 20:
                    img = cv2.imread(self.img_path[i][j], cv2.IMREAD_COLOR)
                    b, g, r = cv2.split(img)
                    img = cv2.merge([r, g, b])
                    img = cv2.resize(img,(224,224),cv2.INTER_LINEAR)
                    img_resize.append(img)
                
                # gps 좌표와 속력을 읽어온 뒤 UTM 좌표와 속력으로 이루어진 시퀀스로 저장
                num = self.img_path[i][j].split('.')[1].split('/')[-1]
                gps_path = self.root + self.img_path[i][j].split('/')[-3] + '/gps/' + str(num) + '.txt'
                gps = open(gps_path, 'r')
                gps.readline()
                gps_data = gps.readline().split(',')
                
                utm_tuple = utm.from_latlon(float(gps_data[1]),float(gps_data[2]))
                utm_list = [np.float64(x) for x in utm_tuple[0:2]]
                self.utm_pos.append(utm_list)
                
                # 처음 좌표값으로 offset
                if j==0:
                    ix_utm=self.utm_pos[j][0]
                    iy_utm=self.utm_pos[j][1]
                    self.utm_pos[j][0]=0.
                    self.utm_pos[j][1]=0.
                else:
                    self.utm_pos[j][0]=self.utm_pos[j][0]-ix_utm
                    self.utm_pos[j][1]=self.utm_pos[j][1]-iy_utm
                self.utm_pos[j].append(np.sqrt(np.square(float(gps_data[10]))+np.square(float(gps_data[11]))+np.square(float(gps_data[12]))))
            
            # states.txt 파일에서 state값을 읽어 저장
            state_path = self.root + self.img_path[i][j].split('/')[-3] + '/states.txt'
            state = open(state_path, 'r')
            state_data = state.readline().split()[9:-20]
            
            utm_np = np.array(self.utm_pos)
            # input 데이터 시퀀스
            for k in range(len(self.img_path[i])-20):
                if k>=9:
                    self.img_seq.append(img_resize[k-9:k+1])
                    # 시퀀스별 현재 사진 기준 offset 및 회전변환을 통한 +y축 방향 설정
                    self.pos_seq.append(self.rotate(self.coord_sub(utm_np[k-9:k+1],utm_np[k]), utm_np[k+1][0]-utm_np[k][0], utm_np[k+1][1]-utm_np[k][1]))
                    self.gt_seq.append(self.rotate(self.coord_sub(utm_np[k+1:k+21],utm_np[k]), utm_np[k+1][0]-utm_np[k][0], utm_np[k+1][1]-utm_np[k][1]))
            self.states = np.append(self.states, state_data)
    
    # 속력을 제외하고 좌표만 offset하기 위한 함수
    def coord_sub(self, seq1, sub):
        return np.c_[seq1[...,:2]-sub[:2],seq1[...,2]]
    
    # 회전변환을 위한 함수
    def rotate(self, seq, dx ,dy):
        angle = np.arctan2(dy,dx)
        for i in range(len(seq)):
            x = seq[i][0]
            y = seq[i][1]
            seq[i][0] = np.cos(np.pi/2-angle)*x - np.sin(np.pi/2-angle)*y
            seq[i][1] = np.sin(np.pi/2-angle)*x + np.cos(np.pi/2-angle)*y
        return seq
            
    def __len__(self):
        return len(self.img_seq)

    def __getitem__(self, idx):
        
        new_img_seq = torch.FloatTensor(np.array(self.img_seq[idx]))
        
        new_pos_seq = torch.FloatTensor(np.array(self.pos_seq[idx]))
        new_gt_seq = torch.FloatTensor(np.array(self.gt_seq[idx]))
        
        if self.states[idx] == 'straight':
            new_state = torch.Tensor([0])
        elif self.states[idx] == 'left':
            new_state = torch.Tensor([1])
        else:
            new_state = torch.Tensor([-1])
            
        return new_img_seq, new_pos_seq, new_gt_seq, new_state

class Trainer(object):
    def __init__(self, epochs, batch_size, lr):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = lr
        self._build_model()

        dataset = LoadDataset(method='train')
        self.root = dataset.root
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.datalen = dataset.__len__()

        # pretrained_weight 로드
        self.weight_root = self.root.split('/')
        del self.weight_root[-2]
        # 아래 경로를 pretrained weight 경로로 수정
        self.weight_root = "/".join(self.weight_root) + 'weight/' + '_210222_modstate_20_model'
        weight_PATH_st = self.weight_root + '_st.pth'
        weight_PATH_left = self.weight_root + '_left.pth'
        weight_PATH_right = self.weight_root + '_right.pth'
#        self.cnnlstmNet.load_state_dict(torch.load(weight_PATH_st))
#        self.cnnlstmleftNet.load_state_dict(torch.load(weight_PATH_left))
#        self.cnnlstmrightNet.load_state_dict(torch.load(weight_PATH_right))
        # weight 저장을 위한 weight_root 재설정
        self.weight_root= self.weight_root.split('/')
        del self.weight_root[-1]
        self.weight_root = "/".join(self.weight_root) + '/'
        
        print("Training...")

    def _build_model(self):
        cnnlstmNet = CNN_LSTM()
        cnnlstmleftNet = CNN_LSTM_LEFT()
        cnnlstmrightNet = CNN_LSTM_RIGHT()
        cnnlstmNet = nn.DataParallel(cnnlstmNet)
        cnnlstmleftNet = nn.DataParallel(cnnlstmleftNet)
        cnnlstmrightNet = nn.DataParallel(cnnlstmrightNet)
        self.cnnlstmNet = cnnlstmNet.to(device)
        self.cnnlstmleftNet = cnnlstmleftNet.to(device)
        self.cnnlstmrightNet = cnnlstmrightNet.to(device)
        self.cnnlstmNet.train()
        self.cnnlstmleftNet.train()
        self.cnnlstmrightNet.train()

        print('Finish build model.')
    
    def train(self):
        # 원하는 이름으로 수정
        date = '210223_adaptstate'
        # epoch당 Average error를 저장하는 list
        self.errors = []
        # Adam과 MSELoss(sum)를 사용
        optimizer = torch.optim.Adam(self.cnnlstmNet.parameters(), lr=self.learning_rate)
        optimizer_left = torch.optim.Adam(self.cnnlstmleftNet.parameters(), lr=self.learning_rate)
        optimizer_right = torch.optim.Adam(self.cnnlstmrightNet.parameters(), lr=self.learning_rate)
        loss = torch.nn.MSELoss(reduction='sum')
        for epoch in tqdm.tqdm(range(self.epochs)):
            # error 확인용 변수
            epoch_error = 0
            sterr = 0
            lefterr = 0
            righterr = 0
            stcount = 0
            leftcount = 0
            rightcount = 0
            
            # 특정 epoch마다 weight 저장
            if epoch % 1 == 0:
                torch.save(self.cnnlstmNet.state_dict(), "_".join([self.weight_root, date, str(epoch+1), 'model_st.pth']))
                torch.save(self.cnnlstmleftNet.state_dict(), "_".join([self.weight_root, date, str(epoch+1), 'model_left.pth']))
                torch.save(self.cnnlstmrightNet.state_dict(), "_".join([self.weight_root, date, str(epoch+1), 'model_right.pth']))
                
            for batch_idx, samples in enumerate(self.dataloader):
                batch_error = 0
                error_total = 0
                img_train, pos_train, gt_train, state = samples
                # state별 인덱스
                state = state.view(-1)
                st = torch.where(state==0)
                left = torch.where(state==1)
                right = torch.where(state==-1)
                
                # straight, left, right에 따라 각각의 network를 통해 나온 loss값으로 역전파 진행
                if len(st[0]) != 0:
                    self.pred_pos_st = self.cnnlstmNet(img_train.cuda()[st],pos_train.cuda()[st])
                    self.gt_pos_st = gt_train[st]
                    optimizer.zero_grad()
                    error_st = loss(self.pred_pos_st.cpu(), self.gt_pos_st)
                    sterr += error_st.data
                    stcount += len(st[0])
                    error_total += error_st.data
                    error_st.backward(retain_graph = True)
                if len(left[0]) != 0:
                    self.pred_pos_left = self.cnnlstmleftNet(img_train.cuda()[left],pos_train.cuda()[left])
                    self.gt_pos_left = gt_train[left]
                    optimizer_left.zero_grad()
                    error_left = loss(self.pred_pos_left.cpu(), self.gt_pos_left)
                    lefterr += error_left.data
                    leftcount += len(left[0])
                    error_total += error_left.data
                    error_left.backward(retain_graph = True)
                if len(right[0]) != 0:
                    self.pred_pos_right = self.cnnlstmrightNet(img_train.cuda()[right],pos_train.cuda()[right])
                    self.gt_pos_right = gt_train[right]
                    optimizer_right.zero_grad()
                    error_right = loss(self.pred_pos_right.cpu(), self.gt_pos_right)
                    righterr += error_right.data
                    rightcount += len(right[0])
                    error_total += error_right.data
                    error_right.backward(retain_graph = True)
                
                batch_error += error_total
                
                # parameter 업데이트
                if len(st[0]) != 0:
                    optimizer.step()
                if len(left[0]) != 0:
                    optimizer_left.step()
                if len(right[0]) != 0:
                    optimizer_right.step()
                epoch_error += batch_error
            self.errors.append(epoch_error/self.datalen)
            # loss 표시용 - 각 state별 갯수와 error를 볼 때 주석 해제
            #print(sterr, stcount, lefterr, leftcount, righterr, rightcount, epoch_error, self.datalen)
            if stcount != 0:
                print('Error of straight in epoch %d: %.4f' %(epoch+1, sterr/stcount))
            if leftcount != 0:
                print('Error of left in epoch %d: %.4f' %(epoch+1, lefterr/leftcount))
            if rightcount != 0:
                print('Error of right in epoch %d: %.4f' %(epoch+1, righterr/rightcount))
            print('Error in epoch %d: %.4f' %(epoch+1, epoch_error/self.datalen))
        print('Finish training.')
        plt.figure(1)
        plt.suptitle('Average error of each epoch')
        plt.plot(self.errors)

class Tester(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self._build_model()

        self.dataset = LoadDataset(method='test')
        self.root = self.dataset.root
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.datalen = self.dataset.__len__()
        self.mse_all_img = []

        # weight 로드
        self.root = './RF_0/'
        self.weight_root = self.root.split('/')
        # 아래 경로를 weight 경로로 수정
        self.weight_root = "/".join(self.weight_root) + 'weight/'
        weight_PATH_st = self.weight_root + 'pretrained_model' + '_st.pth'
        weight_PATH_left = self.weight_root + 'pretrained_model' + '_left.pth'
        weight_PATH_right = self.weight_root + 'pretrained_model' + '_right.pth'
        
        # pretrained_weight가 있을 경우 주석 해제후 로드
        self.cnnlstmNet.load_state_dict(torch.load(weight_PATH_st))
        self.cnnlstmleftNet.load_state_dict(torch.load(weight_PATH_left))
        self.cnnlstmrightNet.load_state_dict(torch.load(weight_PATH_right))
        
        print("Testing...")

    def _build_model(self):
        cnnlstmNet = CNN_LSTM()
        cnnlstmleftNet = CNN_LSTM_LEFT()
        cnnlstmrightNet = CNN_LSTM_RIGHT()
        cnnlstmNet = nn.DataParallel(cnnlstmNet)
        cnnlstmleftNet = nn.DataParallel(cnnlstmleftNet)
        cnnlstmrightNet = nn.DataParallel(cnnlstmrightNet)
        self.cnnlstmNet = cnnlstmNet.to(device)
        self.cnnlstmleftNet = cnnlstmleftNet.to(device)
        self.cnnlstmrightNet = cnnlstmrightNet.to(device)
        self.cnnlstmNet.eval()
        self.cnnlstmleftNet.eval()
        self.cnnlstmrightNet.eval()
        print('Finish build model.')
    
    def test(self):
        # error 확인용 변수
        total_error_pos = 0.
        total_error_pos_st = 0.
        total_error_pos_left = 0.
        total_error_pos_right = 0.
        total_error_speed = 0.
        total_error_speed_st = 0.
        total_error_speed_left = 0.
        total_error_speed_right = 0.
        errors = []
        errors_pos = []
        errors_speed = []
        count_st = 0
        count_left = 0
        count_right = 0
        for batch_idx, samples in enumerate(self.dataloader):
            bat_error_pos = 0.
            bat_error_speed = 0.
            img_test, self.pos_test, self.gt_test, state = samples
            state = state.view(-1)
            # state별 인덱스
            st = torch.where(state==0)
            left = torch.where(state==1)
            right = torch.where(state==-1)
            count_st += len(st[0])
            count_left += len(left[0])
            count_right += len(right[0])
            with torch.no_grad():
                if len(st[0]) != 0:
                    self.plan_pos_st = self.cnnlstmNet(img_test.cuda()[st],self.pos_test.cuda()[st])
                if len(left[0]) != 0:
                    self.plan_pos_left = self.cnnlstmNet(img_test.cuda()[left],self.pos_test.cuda()[left])
                if len(right[0]) != 0:
                    self.plan_pos_right = self.cnnlstmNet(img_test.cuda()[right],self.pos_test.cuda()[right])
            self.gt_pos_st = self.gt_test[st]
            self.gt_pos_left = self.gt_test[left]
            self.gt_pos_right = self.gt_test[right]
            
            loss = torch.nn.MSELoss(reduction='sum')
            
            # state별 오차 합 계산
            if len(st[0]) != 0:
                bat_error_pos_st = loss(self.plan_pos_st.cpu()[:,:,:2], self.gt_pos_st[:,:,:2])
                bat_error_speed_st = loss(self.plan_pos_st.cpu()[:,:,2], self.gt_pos_st[:,:,2])
                bat_error_pos += bat_error_pos_st.data
                bat_error_speed += bat_error_speed_st.data
                total_error_pos_st += bat_error_pos_st.data
                total_error_speed_st += bat_error_speed_st.data
            if len(left[0]) != 0:
                bat_error_pos_left = loss(self.plan_pos_left.cpu()[:,:,:2], self.gt_pos_left[:,:,:2])
                bat_error_speed_left = loss(self.plan_pos_left.cpu()[:,:,2], self.gt_pos_left[:,:,2])
                bat_error_pos += bat_error_pos_left.data
                bat_error_speed += bat_error_speed_left.data
                total_error_pos_left += bat_error_pos_left.data
                total_error_speed_left += bat_error_speed_left.data
            if len(right[0]) != 0:
                bat_error_pos_right = loss(self.plan_pos_right.cpu()[:,:,:2], self.gt_pos_right[:,:,:2])
                bat_error_speed_right = loss(self.plan_pos_right.cpu()[:,:,2], self.gt_pos_right[:,:,2])
                bat_error_pos += bat_error_pos_right.data
                bat_error_speed += bat_error_speed_right.data
                total_error_pos_right += bat_error_pos_right.data
                total_error_speed_right += bat_error_speed_right.data
            
            total_error_pos += bat_error_pos
            total_error_speed += bat_error_speed
            
            errors_pos.append(bat_error_pos/self.batch_size)
            errors_speed.append(bat_error_speed/self.batch_size)
            #print("total error: %.4f" % total_error)
#         plt.figure(1)
#         plt.suptitle('Average displacement error')
#         plt.plot(errors_pos)
#         plot1 = plt.figure(2)
#         plt.suptitle('Average speed error')
#         plt.plot(errors_speed)
        # state별 error 확인용 - 불필요시 주석처리
        print('avg total error: %.4f' %((total_error_pos+total_error_speed)/self.datalen))
        print('avg total displacement error: %.4f' %(total_error_pos/self.datalen))
        print('avg total speed error: %.4f' %(total_error_speed/self.datalen))
        print()
        if len(st[0]) != 0:
            print('avg straight error: %.4f' %((total_error_pos_st+total_error_speed_st)/count_st))
            print('avg straight displacement error: %.4f' %(total_error_pos_st/count_st))
            print('avg straight speed error: %.4f' %(total_error_speed_st/count_st))
        print()
        if len(left[0]) != 0:
            print('avg left error: %.4f' %((total_error_pos_left+total_error_speed_left)/count_left))
            print('avg left displacement error: %.4f' %(total_error_pos_left/count_left))
            print('avg left speed error: %.4f' %(total_error_speed_left/count_left))
        print()
        if len(right[0]) != 0:
            print('avg right error: %.4f' %((total_error_pos_right+total_error_speed_right)/count_right))
            print('avg right displacement error: %.4f' %(total_error_pos_right/count_right))
            print('avg right speed error: %.4f' %(total_error_speed_right/count_right))
        print()
    # position, speed plot용 함수
    def view(self, img_seq, pos_seq, gt_seq, state, viewtype):
        with torch.no_grad():
            if state==0:
                plan_pos = self.cnnlstmNet(img_seq.cuda(),pos_seq.cuda())
            if state==1:
                plan_pos = self.cnnlstmleftNet(img_seq.cuda(),pos_seq.cuda())
            if state==-1:
                plan_pos = self.cnnlstmrightNet(img_seq.cuda(),pos_seq.cuda())
        gt_pos = gt_seq
        
        x = []
        y = []
        v = []
        x_p = []
        y_p = []
        v_p = []
        x_g = []
        y_g = []
        v_g = []
        
        for i in range(10):
            x.append(pos_seq[0][i][0])
            y.append(pos_seq[0][i][1])
            v.append(pos_seq[0][i][2])
            v_p.append(None)
            v_g.append(None)
        for i in range(20):
            x_p.append(plan_pos[0][i][0])
            y_p.append(plan_pos[0][i][1])
            v_p.append(plan_pos[0][i][2])
            x_g.append(gt_pos[0][i][0])
            y_g.append(gt_pos[0][i][1])
            v_g.append(gt_pos[0][i][2])
        
        # 범례 없이 graph만 시각화
        if viewtype == 'pos':
            plt.plot(x, y, label='Past Path')
            plt.plot(x_p, y_p, label='Predicted Path')
            plt.plot(x_g, y_g, label='GT Path')
        if viewtype == 'speed':
            plt.plot(v, label='Past Speed')
            plt.plot(v_p, label='Predicted Speed')
            plt.plot(v_g, label='GT Speed')
        
        # 범례 포함 graph 시각화
        if viewtype == 'pos_l':
            plt.plot(x, y, label='Past Path')
            plt.plot(x_p, y_p, label='Predicted Path')
            plt.plot(x_g, y_g, label='GT Path')
            plt.legend()
        if viewtype == 'speed_l':
            plt.plot(v, label='Past Speed')
            plt.plot(v_p, label='Predicted Speed')
            plt.plot(v_g, label='GT Speed')
            plt.legend()
    
    # 이미지 - position - speed 순 그래프 시각화
    def visualize(self, idx, pos_axes = [-20, 20, -20, 100], speed_axes = [0, 30, 0, 30], pos_vtype = 'pos_l', speed_vtype = 'speed_l'):
        img, pos, gt, state = self.dataset.__getitem__(idx)
        img = torch.reshape(img,(1,img.shape[0],img.shape[1],img.shape[2],img.shape[3]))
        pos = torch.reshape(pos,(1,pos.shape[0],pos.shape[1]))
        gt = torch.reshape(gt,(1,gt.shape[0],gt.shape[1]))
        state = torch.reshape(state,(1,1))

        plot1 = plt.figure(0)
        plt.axis('off')
        plt.imshow(img[0][9].type(torch.IntTensor))
        plot2 = plt.figure(1)
        plt.axis(pos_axes)
        self.view(img,pos,gt,state,pos_vtype)
        plot3 = plt.figure(2)
        plt.axis(speed_axes)
        self.view(img,pos,gt,state,speed_vtype)

# states.txt 임의 생성 함수
def MakeStates(method=None):
    # 데이터셋 경로에 따라 아래 경로 수정
    root = './RF_0/'

    temp = np.array([])

    if method == 'train':       
        root = root + 'train/'

    elif method == 'test': 
        root = root + 'test/'

    elif method == 'temp': 
        root = root + 'temp/'

    num_dir = glob(root + '*')
    sort_dir = sorted(num_dir, key = lambda j : int(os.path.splitext(os.path.basename(j))[0]))
    img_path=[]

    for i in tqdm.tqdm(range(len(sort_dir))):
        temp = []
        path = glob(sort_dir[i] + '/image1/*.png')
        sort_path = sorted(path,key = lambda j : int(os.path.splitext(os.path.basename(j))[0]))
        img_path.append(sort_path)
        utm_pos = []
        img_resize = []
        isTurning = False
        willTurn = False
        direction = 'straight'
        idx = 0
        temp_speed = 0
        temp_idx = 0

        for j in range(len(img_path[i])):


            num = img_path[i][j].split('.')[1].split('/')[-1]

            gps_path = root + img_path[i][j].split('/')[-3] + '/gps/' + str(num) + '.txt'
            gps = open(gps_path, 'r')
            gps.readline()
            gps_data = gps.readline().split(',')

            speed = np.sqrt(np.square(float(gps_data[10]))+np.square(float(gps_data[11]))+np.square(float(gps_data[12])))

            can_path = root + img_path[i][j].split('/')[-3] + '/CAN/' + str(num) + '.txt'
            can = open(can_path, 'r')
            can.readline()
            can_data = can.readline().split(',')
            sas = float(can_data[0])

            # 조향각이 ±60도를 넘을 경우 회전 시작으로 판단 후 해당 인덱스 저장
            if abs(sas)>60 and willTurn == False:
                willTurn = True
                temp_idx = j
                temp_speed = speed

            # 조향각이 ±120도를 넘지 않고 다시 돌아올 경우 회전으로 판단하지 않음
            if abs(sas) < 5 and willTurn == True and isTurning == False:
                willTurn = False

            # 조향각이 ±120도를 넘은 경우 회전으로 판단 및 회전 방향 설정
            if sas>120 and isTurning == False:
                isTurning = True
                direction = 'left'
            elif sas<-120 and isTurning == False:
                isTurning = True
                direction = 'right'

            # 조향각이 ±5도 미만이 될 경우(회전을 마쳤다고 판단) 해당 지점에서 회전 시작지점 이전 30장의 이미지에 회전 방향 state 저장
            if abs(sas) < 5 and isTurning == True:
                start_dist = int(30*5/temp_speed)
                if direction == 'left':
                    if temp_idx < start_dist:
                        temp[0:j] = 'left'
                    else:
                        temp[temp_idx-start_dist:j] = 'left'
                    isTurning = False
                    willTurn = False
                    idx = j
                    direction == 'straight'
                elif direction == 'right':
                    if temp_idx < start_dist:
                        temp[0:j] = 'right'
                    else:
                        temp[temp_idx-start_dist:j] = 'right'
                    isTurning = False
                    willTurn = False
                    idx = j
                    direction == 'straight'

            temp = np.append(temp,'straight')

            # 해당 폴더 내부에 state.txt 형태로 이미지별 state 저장
            if j == len(img_path[i])-1:
                state_path = root + img_path[i][j].split('/')[-3] + '/states.txt'
                np.savetxt(state_path,temp,fmt='%s',newline=" ")

def main():
    epochs = 100
    batchSize = 32
    learningRate = 1e-3
    # MakeStates는 한 번 실행한 이후는 주석처리 가능
    MakeStates(method='train')
    MakeStates(method='test')
    
    #trainer = Trainer(epochs, batchSize, learningRate)
    #trainer.train()

    tester = Tester(batchSize)
    tester.test()
    tester.visualize(10)

if __name__ == '__main__':
    main()
