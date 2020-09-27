from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot



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
    
    mythr=0.5
    for i in range (18):
      if(len(result[i])>0):
        for j in range(len(result[i])):
          if(result[i][j][4]>mythr):
            print(result[i][j])
        
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
    
    # show the results
    show_result_pyplot(model, args.img, result, score_thr=args.score_thr)


if __name__ == '__main__':
    main()
