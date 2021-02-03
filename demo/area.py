from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import numpy as np
import cv2


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # test a single image
    result = inference_detector(model, args.img)
    
    # show the results
    box=show_result_pyplot(model, args.img, result, score_thr=args.score_thr)#k=shape of img
    k=box.shape
    
    
    
    #####
    # img :original pic
    # box :pic with bounding box
    img = cv2.imread(args.img)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(10,10)) # clipLimit
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cl = clahe.apply(gray)
    blurred = cv2.GaussianBlur(cl, (7, 7), 0)
    canny = cv2.Canny(blurred, 40, 120)
    ret, thresh = cv2.threshold(canny, 127, 255, cv2.THRESH_BINARY)
    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    width = img.shape[1]
    height = img.shape[0] 
    print('width='+str(width)+' height='+str(height))
    canvas = np.zeros((height, width), np.uint8)
    cv2.drawContours(canvas, cnts, -1, (255, 255, 255), 1)
    #####
    
    white = np.sum(canvas == 255)
    print('total contour:'+str(white))

    ###
    row, column = k[0], k[1] 
    Array= [[0 for _ in range(row)] for _ in range(column)]
    #print(Array[700][500])
    
    ###
    mythr=0.3
    for i in range (18): #i:class
      if(i==17):
        sumarea=0
        overlap=0
        if(len(result[i])>0):
          for j in range(len(result[i])):
            if(result[i][j][4]>mythr):
              x1=result[i][j][0]
              y1=result[i][j][1]
              x2=result[i][j][2]
              y2=result[i][j][3]
              
              ######
              #print(x1,x2,y1,y2)  #51.707787 677.9785 25.907877 442.26596
              #print(int(x1+0.5),int(x2+0.5),int(y1+0.5),int(y2+0.5))
              
              for x in range(int(x1+0.5),int(x2+0.5)):
                for y in range (int(y1+0.5),int(y2+0.5)):
                  if(canvas[y][x]==255 and Array[x][y]==0):
                    Array[x][y]=1
                    sumarea=sumarea+1
                  else:
                    overlap=overlap+1
              #print(str(j)+'area:'+str(sumarea))  
        print('contourarea of class'+str(i)+': '+str(sumarea))
        print('ratio:'+str((sumarea)/(k[0]*k[1])))
      else:
        sumarea=0
        overlap=0
        if(len(result[i])>0):
          for j in range(len(result[i])):
            if(result[i][j][4]>mythr):
              x1=result[i][j][0]
              y1=result[i][j][1]
              x2=result[i][j][2]
              y2=result[i][j][3]
              
              ######
              #print(x1,x2,y1,y2)  #51.707787 677.9785 25.907877 442.26596
              #print(int(x1+0.5),int(x2+0.5),int(y1+0.5),int(y2+0.5))
              for x in range(int(x1+0.5),int(x2+0.5)):
                for y in range (int(y1+0.5),int(y2+0.5)):
                  if(Array[x][y]==0):
                    Array[x][y]=1
                    sumarea=sumarea+1
                  else:
                    overlap=overlap+1  
        print('sumarea of class'+str(i)+': '+str(sumarea))
        #print('overlap of class'+str(i)+': '+str(overlap))
        print('ratio:'+str((sumarea)/(k[0]*k[1])))  
    
    #show imgs
    cv2.namedWindow('show', cv2.WINDOW_NORMAL)
    c=cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    imgs = np.hstack([box,c])
    cv2.imshow('show',imgs)
    cv2.waitKey(0)
    
if __name__ == '__main__':
    main()
