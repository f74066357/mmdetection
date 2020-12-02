from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import numpy as np


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
    k=show_result_pyplot(model, args.img, result, score_thr=args.score_thr)#k=shape of img
    
    ###
    row, column = k[0], k[1] 
    Array= [[0 for _ in range(row)] for _ in range(column)]
    #print(Array[700][500])
    
    ###
    mythr=0.5
    for i in range (18): #i:class
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

            for x in range(int(x1),int(x2)):
              for y in range (int(y1),int(y2)):
                if(Array[x][y]==0):
                  Array[x][y]=1
                  sumarea=sumarea+1
                else:
                  overlap=overlap+1
            
            ########
            #print((result[17]))
            '''
            w=abs(x2-x1)
            h=abs(y2-y1)
            print("here")
            print('h:'+str(k[0])+' w:'+str(k[1])) #shape
            print('range h:'+str(w)+' range w:'+str(h))
            #ratio
            print((h*w)/(k[0]*k[1]))
            '''
  #####
      print('sumarea of class'+str(i)+': '+str(sumarea))
      #print('overlap of class'+str(i)+': '+str(overlap))
      print('ratio:'+str((sumarea)/(k[0]*k[1])))
  #####
            
    #print(w,h,w*h)
    #print(len(result[0]))
    #print(len(result[17]))
    #print((result[17]))
    
    #result[1][0][0]=100
    #result[1][0][1]=500
    #result[1][0][2]=800
    #result[1][0][3]=200
    #print(result[1][0][4])
    #l u r d

    #k=[[923.4591    332.5638    100.5444    403.03183     0.9977043]]
    
    ###


if __name__ == '__main__':
    main()
