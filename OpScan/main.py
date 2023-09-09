from turtle import width
import cv2
from cv2 import split
import numpy as np
from utlis import utlis
##############
path="C:\\Users\\beyza\\Desktop\\proje\\isaretlenmis_tam.jpg"
widthImg=700
heightImg=1400
questions=20
choices=5
#dogru cevap anahtarı
ans=[0, 3, 2, 1, 4, 4, 0, 1, 2, 3, 3, 2, 0, 1, 0, 2, 0, 3, 0, 1]
ans1=[0, 2, 2, 1, 4, 4, 0, 1, 2, 3, 3, 2, 0, 1, 0, 2, 0, 3, 0, 1]

webcamFeed=True
cameraNo=0

cap=cv2.VideoCapture(cameraNo)
cap.set(10,150)

while True:
    if webcamFeed: success, img=cap.read()
    else :img=cv2.imread("C:\\Users\\beyza\\Desktop\\proje\\isaretlenmis_tam.jpg")
        #PROCESSING
    
    img=cv2.resize(img,(widthImg,heightImg))
    
    
        #imgContours=img.copy()
    imgFinal=img.copy()
    imgFinal2=img.copy()
    imgToplam=img.copy()


    imgBiggestContours=img.copy()
    #imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #imgBlur=cv2.GaussianBlur(imgGray,(5,5),1)
    #imgCanny=cv2.Canny(imgBlur,10,100)
    imgCanny=cv2.Canny(img,10,100)
    try:    
            #FINDING ALL CONTOURS
        contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            #cv2.drawContours(imgContours,contours,-1,(0,255,0),1)
            #FIND RECTANGLES
        rectCon=utlis.rectContour(contours)
                #print(rectCon)
        biggestContour=utlis.getCornerPoints(rectCon[0])
        biggestContour1=utlis.getCornerPoints(rectCon[1])
        biggestContour2=utlis.getCornerPoints(rectCon[2])



        gradePoints=utlis.getCornerPoints(rectCon[2])

        print(biggestContour1)

        if biggestContour.size!=0 and gradePoints.size!=0:
            #biggestContour->grade kutusu 
            cv2.drawContours(imgBiggestContours,biggestContour,-1,(0,255,0),20)
            cv2.drawContours(imgBiggestContours,biggestContour1,-1,(0,255,128),20)
            cv2.drawContours(imgBiggestContours,biggestContour2,-1,(0,128,128),20)

            #Optikte sonucun gösterdiği yer gradePoints
            cv2.drawContours(imgBiggestContours,gradePoints,-1,(255,0,0),20)
            biggestContour=utlis.reorder(biggestContour)
            biggestContour1=utlis.reorder(biggestContour1)
            biggestContour2=utlis.reorder(biggestContour2)
            
            gradePoints=utlis.reorder(gradePoints)
                    
            pt1=np.float32(biggestContour)
            pt3=np.float32(biggestContour1)
            pt4=np.float32(biggestContour2)
                    
            pt2=np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
            matrix=cv2.getPerspectiveTransform(pt3,pt2)
            matrix1=cv2.getPerspectiveTransform(pt3,pt2)
            matrix2=cv2.getPerspectiveTransform(pt1,pt2)
                    #1.kutu
            imgWarpColored=cv2.warpPerspective(img,matrix,(widthImg,heightImg))
                    #3.kutu
            imgWarpColored1=cv2.warpPerspective(img,matrix1,(widthImg,heightImg))
                    #2.kutu
            imgWarpColored2=cv2.warpPerspective(img,matrix2,(widthImg,heightImg))
                    #grade
            ptG1=np.float32(gradePoints)
            ptG2=np.float32([[0,0],[325,0],[0,150],[325,150]])
            matrixG=cv2.getPerspectiveTransform(ptG1,ptG2)
            imgGradeDisplay=cv2.warpPerspective(img,matrixG,(325,150))
            #cv2.imshow("Grade",imgGradeDisplay)

                    #APPLY THRESHOLD
            imgWarpGray=cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
            imgWarpGray2=cv2.cvtColor(imgWarpColored2,cv2.COLOR_BGR2GRAY)

            imgThresh=cv2.threshold(imgWarpGray,170,255,cv2.THRESH_BINARY_INV)[1]
            imgThresh2=cv2.threshold(imgWarpGray2,170,255,cv2.THRESH_BINARY_INV)[1]

                    #satır almak için:
            boxes=utlis.splitBoxes(imgThresh)
                    #cv2.imshow("Test",boxes[2])
                    #print(cv2.countNonZero(boxes[0]),cv2.countNonZero(boxes[1]))
                    # EN YÜKSEK İNDEX(SAYI)'YA SAHİP SAYIYI KARALANMIŞ SAYAR
            myPixelVal=np.zeros((questions,choices))
                    #countRow and countColumn
            countC=0
            countR=0
            for image1 in boxes:
                totalPixels=cv2.countNonZero(image1)
                myPixelVal[countR][countC]=totalPixels
                countC+=1
                if(countC==choices):countR+=1;countC=0
            print(myPixelVal)
        #*******************************
            boxes1=utlis.splitBoxes(imgThresh2)
                    #cv2.imshow("Test",boxes[2])
                    #print(cv2.countNonZero(boxes[0]),cv2.countNonZero(boxes[1]))
                    # EN YÜKSEK İNDEX(SAYI)'YA SAHİP SAYIYI KARALANMIŞ SAYAR
            myPixelVal1=np.zeros((questions,choices))
                    #countRow and countColumn
            countC=0
            countR=0
            for image in boxes1:
                totalPixels=cv2.countNonZero(image)
                myPixelVal1[countR][countC]=totalPixels
                countC+=1
                if(countC==choices):countR+=1;countC=0
            print(myPixelVal)

                    #FINDING INDEX VALUES OF THE MARKINGS
                #hangi arraydeki index büyükse onu işaretlenen şık algılıyo
            myIndex=[]
            for x in range (0,questions):
                arr=myPixelVal[x]
                        #print("arr",arr)
                        #myIndexVal=np.where(arr==np.amax(arr))
                isNull = True
                for i in range(5):
                    if arr[i] >= 4500:
                        myIndex.append(i)
                        isNull = False
                if isNull:  
                    myIndex.append(-1)
                        #print(myIndexVal[0])
                        #arrayları listeler
                        #myIndex.append(myIndexVal[0][0])
                    #print(myIndex)
            myIndex1=[]
            for x in range (0,questions):
                arr=myPixelVal1[x]
                        #print("arr",arr)
                        #myIndexVal=np.where(arr==np.amax(arr))
                isNull = True
                for i in range(5):
                    if arr[i] >= 4500:
                        myIndex1.append(i)
                        isNull = False
                if isNull:  
                    myIndex1.append(-1)
                        #print(myIndexVal[0])
                        #arrayları listeler
                        #myIndex.append(myIndexVal[0][0])
                    #print(myIndex)

                    #GRADING,not hesaplama
                
            grading=[]
            for x in range (0,questions):
                if ans[x]==myIndex[x]:
                    grading.append(1)
                else: grading.append(0)
                    #print(grading)
            score=(sum(grading)/questions)*100 #FINAL GRADE
            print(score)

            grading1=[]
            for x in range (0,questions):
                if ans1[x]==myIndex1[x]:
                    grading1.append(1)
                else: grading1.append(0)
                    #print(grading)
            score1=(sum(grading1)/questions)*100 #FINAL GRADE
            print(score)
            score2=(score1+score)/2
            print(score2)
                
                    #DISPLAYING ANSWERS
            imgResult2=imgWarpColored2.copy()
            imgResult=imgWarpColored.copy()
            print("test: "+str(imgResult.shape))
            imgResult=utlis.showAnswers(imgResult,myIndex,grading,ans,questions,choices)
            imgResult2=utlis.showAnswers(imgResult2,myIndex1,grading1,ans1,questions,choices)


        #1.kutunun doğru yanlış gösterimi
            imRawDrawing= np.zeros_like(imgWarpColored)
            imRawDrawing=utlis.showAnswers(imRawDrawing,myIndex,grading,ans,questions,choices)
            invMatrix=cv2.getPerspectiveTransform(pt2,pt3)
            imgInvWarp=cv2.warpPerspective(imRawDrawing,invMatrix,(widthImg,heightImg))
        #2.kutunun doğru yanlış gösterimi
            imRawDrawing2=np.zeros_like(imgWarpColored2)
            imRawDrawing2=utlis.showAnswers(imRawDrawing2,myIndex1,grading1,ans1,questions,choices)
            invMatrix2=cv2.getPerspectiveTransform(pt2,pt1)
            imgInvWarp2=cv2.warpPerspective(imRawDrawing2,invMatrix2,(widthImg,heightImg))


            imgRawGrade=np.zeros_like(imgGradeDisplay)
            cv2.putText(imgRawGrade,str(float(score2))+"",(60,100),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,128),3)
            #cv2.imshow("Grade",imgRawGrade)
            invMatrixG=cv2.getPerspectiveTransform(ptG2,ptG1)
            imgInvGradeDisplay=cv2.warpPerspective(imgRawGrade,invMatrixG,(widthImg,heightImg))
        #    cv2.imshow("Gradee",imgInvGradeDisplay)

            imgFinal2=cv2.addWeighted(imgFinal2,1,imgInvWarp2,1,1)
            imgFinal=cv2.addWeighted(imgFinal,1,imgInvWarp,1,0)
            imgFinal=cv2.addWeighted(imgFinal,1,imgInvGradeDisplay,1,0)
            imgToplam=cv2.add(imgFinal,imgFinal2)



            imgBlank=np.zeros_like(img)

            imageArray=([imgBiggestContours,imgWarpGray,imgThresh,imgResult],
                        [imRawDrawing,imgInvWarp,imgFinal,imgInvWarp2])


            """""
                imageArray=([img,imgGray,imgBlur,imgCanny],
                            [imgBiggestContours,imgWarpColored,imgWarpColored2,imgWarpColored1,],
                            [imgContours,imgBlank,imgBlank,imgBlank])

                imageArray=([img,imgContours,imgBiggestContours,imgWarpColored],
                            [imgWarpColored,imgResult,imgWarpColored1,imgThresh,],
                            )
                """
        
        imgBlank=np.zeros_like(img)

        imageArray=([imgBiggestContours,imgResult2,imgToplam,imgResult],
                    [imRawDrawing,imgInvWarp,imgFinal,imgFinal2])
    except:
        imgBlank=np.zeros_like(img)

        imageArray=([imgBlank,imgBlank,imgToplam,imgBlank],
                    [imgBlank,imgBlank,imgBlank,imgBlank])
    lables=[["Big Contour","WarpGray","ToplamSonuc","Result"],
                ["RawDrawing","imgInvWarp", "Final","Bos"]]
    imgStacked=utlis.stackImages(imageArray,0.3,lables)
    #imgStacked2=utlis.stackImages(imageArray(1,[0]),0.9)
#    cv2.imshow("Final Result",imgToplam)
    cv2.imshow("Images",imgStacked)
    #cv2.imshow("En son",imgStacked2)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("FinalResult.jpg",imgToplam)
        cv2.waitKey(300)
        





