import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import Counter
import math
import os

#Binarizacion
def bina(imagen):
    _,bin_img = cv2.threshold(imagen,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bin_img

#Bounding box
def bbox(imagen,row_size_max,col_size_max):
    _, contours, _ = cv2.findContours(imagen, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        rects.append(np.array([x,y,w,h]))
    ### Para evitar que haya rectangulos dentro de otros
    limite_rect_into = 10
    rects_final = []
    for rect in rects:
        x = rect[0]
        y = rect[1]
        xf = x+rect[2]
        yf = y+rect[3]
        cont = 0
        for rect2 in rects:
            x1 = rect2[0]
            y1 = rect2[1]
            xf1 = x1+rect2[2]
            yf1 = y1+rect2[3]
            if (x<x1 and y<y1 and xf1<xf and yf1 < yf):
                cont += 1
            if (cont > limite_rect_into):
                break
        if (cont <= limite_rect_into):
            rects_final.append(rect)
    
    return contours,rects_final

#Lo opuesto
def bbox2(imagen,row_size_max,col_size_max):
    _, contours, _ = cv2.findContours(imagen, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        rects.append(np.array([x,y,w,h]))
    
    rects_final = []
    for rect in rects:
        x = rect[0]
        y = rect[1]
        xf = x+rect[2]
        yf = y+rect[3]
        cont = 0
        for rect2 in rects:
            x1 = rect2[0]
            y1 = rect2[1]
            xf1 = x1+rect2[2]
            yf1 = y1+rect2[3]
            if (x>x1 and y>y1 and xf1>xf and yf1 > yf):
                cont += 1
                break
        if (cont == 0):
            rects_final.append(rect)    
    return rects_final

# Dibujar rectangulos
def print_rect(img,rects,color1='g',grosor=2):
    color = None
    if color1 == 'g':
        color = (0,255,0)
    elif color1 == 'b':
        color = (255,0,0)
    elif color1 == 'r':
        color = (0,0,255)
    else:
        color = (0,255,255)
    for rec in rects:
        x = rec[0]
        y = rec[1]
        w = rec[2]
        h = rec[3]
        cv2.rectangle(img, (x, y), (x+w, y+h), color, grosor)

# Unir bounding boxes
def union(a,b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return (x, y, w, h)

# Detectar interseccion
def intersection(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return () # or (0,0,0,0) ?
    return (x, y, w, h)

# Combinar boxes
def combine_boxes(boxes):
    noIntersectLoop = False
    noIntersectMain = False
    posIndex = 0
     # keep looping until we have completed a full pass over each rectangle
     # and checked it does not overlap with any other rectangle
    while noIntersectMain == False:
        noIntersectMain = True
        posIndex = 0
         # start with the first rectangle in the list, once the first 
         # rectangle has been unioned with every other rectangle,
         # repeat for the second until done
         
        while posIndex < len(boxes):
            noIntersectLoop = False
            while noIntersectLoop == False and len(boxes) > 1 and posIndex<len(boxes):
                a = boxes[posIndex]
                listBoxes = np.delete(boxes, posIndex, 0)
                index = 0
                for b in listBoxes:
                    #if there is an intersection, the boxes overlap
                    if intersection(a, b): 
                        newBox = union(a,b)
                        listBoxes[index] = newBox
                        boxes = listBoxes
                        noIntersectLoop = False
                        noIntersectMain = False
                        index = index + 1
                        break
                    noIntersectLoop = True
                    index = index + 1
            posIndex = posIndex + 1

    return boxes

#Proyecciones
def projection(img,orientacion,rec,porcentaje=1):
    x = rec[0]
    y = rec[1]
    w = rec[2]
    h = rec[3]
    proj = None
    orient = 0
    limite = 0
    extre = 0
    init = 0
    mask = img[y:y+h,x:x+w]
    if (orientacion == 'h'):
        proj = np.asarray(np.sum(mask,axis = 1)).reshape(-1)
        init = y
        limite = w
        extre = y + h
    else:
        proj = np.sum(mask,axis = 0)
        init = x
        limite = h
        extre = x + w
    idx = np.where(proj > limite*porcentaje)[0].tolist()
    if init not in idx:
        idx.insert(0,init)
    if extre not in idx:
        idx.append(extre)
    return idx

#Fragmentar
def fragm(project,orientacion,rect,thresh):
    frags = []
    for i in range(len(project)-1):
        dif = project[i+1] - project[i]
        if (dif > thresh):
            frag = []
            if(orientacion == 'h'):
                frag = [rect[0],project[i],rect[2],dif]
            else:
                frag = [project[i],rect[1],dif,rect[3]]
            frags.append(frag)
    if frags == []:
        frags = rect
    return frags

#Guardar fragmentos
def save_frag(fragments,label,imagen):
    for i in range(len(fragments)):
        subimage = imagen[fragments[i][1]:fragments[i][1]+fragments[i][3],fragments[i][0]:fragments[i][0]+fragments[i][2]]
        cv2.imwrite(label+str(i)+"frag.png",subimage)
#Limpiar fragmentos
def clean_frag(fragments,imagen):
    new_frags = []
    for frag in fragments:
        x = frag[0]
        y = frag[1]
        w = frag[2]
        h = frag[3]
        subimg = imagen[y:y+h,x:x+w]
        subimg = subimg > 0
        if (np.sum(subimg)!=0):
            new_frags.append(frag)
    return new_frags

#Clasificacion
def classifi(LC,LT,LG,frag,f_rects,x_h,cont):
    ix = frag[0]
    iy = frag[1]
    iw = frag[2]
    ih = frag[2]
    cont2 = 0
    #A escala de grises
    aux_img = np.zeros((frag[3],frag[2]))
    for rect in f_rects:
        x = rect[0]
        y = rect[1]
        w = rect[2]
        h = rect[3]
        aux_img[y:y+h,x:x+w] = math.ceil((h/2)%255)
    
    # Dilatacion
    #kernel_dilate = np.ones((math.floor(x_h/3),math.floor(x_h/3)),np.uint8)
    kernel_dilate = np.ones((math.floor(x_h/3),math.floor(x_h*2)),np.uint8)
    ID = cv2.dilate(aux_img,kernel_dilate,iterations=1)
    ID = np.uint8(ID)
    
    #Bounding box
    ID1 = ID.copy()
    rects4 = bbox2(ID1,ih,iw)
    if (len(rects4)>0):
        #Column
        for rect in rects4:
            x = rect[0]
            y = rect[1]
            w = rect[2]
            h = rect[3]
            bounding = ID[y:y+h,x:x+w].copy()
            bounding[bounding <= ((x_h*2)%255) ] = 0
            bounding[bounding > ((x_h*2)%255) ] = 1
            if (np.mean(bounding) < 0.51):
                LC[iy+y:iy+y+h,ix+x:ix+x+w] = 255
                ID[y:y+h,x:x+w] = 0
            
        #Title
        #kernel_dilate = np.ones((math.floor(x_h),math.floor(x_h*3)),np.uint8)
        kernel_dilate = np.ones((math.floor(x_h),math.floor(x_h*2*2)),np.uint8)
        ID2 = cv2.dilate(ID,kernel_dilate,iterations=1)
        ID2 = np.uint8(ID2)
        #cv2.imwrite(folder+str(cont)+"_"+str(cont2)+"ID2.png",ID2)
        ID3_2 = ID2.copy()
        rects5 = bbox2(ID3_2,ih,iw)
        if (len(rects5)>0):
            for rect in rects5:
                x = rect[0]
                y = rect[1]
                w = rect[2]
                h = rect[3]
                if (w > 2*h and h//2 >x_h and w > x_h):
                    LT[iy+y:iy+y+h,ix+x:ix+x+w] = 255
                    ID2[y:y+h,x:x+w] = 0
                elif(w<3*h and h//2 > x_h*2 and w > x_h):
                    LG[iy+y:iy+y+h,ix+x:ix+x+w] = 255
                    ID2[y:y+h,x:x+w] = 0
    return LC,LT,LG

# Asegurse que se tenga textos
def checkbbox(I,rects):
    rects_final = []
    I_aux = I//255
    for rect in rects:
        x = rect[0]
        y = rect[1]
        w = rect[2]
        h = rect[3]
        img_aux = I_aux[y:y+h,x:x+w].copy()
        if(np.mean(img_aux)>0.75):
            rects_final.append(rect)
    return rects_final

# Algoritmo
def alg3(folder,num,write):
    folder = folder+str(num)+"/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    #Abrir la imagen
    img = cv2.imread("imagenes/"+str(num)+".jpg")
    if write:
        cv2.imwrite(folder+"1original.png")
    i = img.copy()

    #Escala de grises
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if write:
        cv2.imwrite(folder+"2gray.png",gray_img)
    row,col = np.shape(gray_img)
    rec_img = [0,0,col,row]

    #Reducir ruido filtro Gaussiano
    gray_img = cv2.GaussianBlur(gray_img,(5,5),0)
    if write:
        cv2.imwrite(folder+"3ruido.png",gray_img)
    
    #Convertir a binario
    bin_img = bina(gray_img)
    if write:
        cv2.imwrite(folder+"4binarioI.png",bin_img)

    #Calcular x-height
    img_a = bin_img.copy()
    contours,rects = bbox(img_a,row,col)
    if write:
        img_print = i.copy()
        print_rect(img_print,rects)
        cv2.drawContours(img_print, contours, -1, (255, 255, 0), 1)
        cv2.imwrite(folder+"5contornos.png",img_print)
    
    #x_height = [(rect[3]/2)for rect in rects]
    x_height = [] 
    for rect in rects:
        x_h_aux = rect[3]/2
        if (x_h_aux>5):
            x_height.append(x_h_aux)
    ocurrencias = Counter(x_height)
    x_h = ocurrencias.most_common(1)[0][0]
    rects1 = []
    for rec in rects:
        if (rec[3]/2 == x_h):
            rects1.append(rec)
    if write:
        i1 = i.copy()
        print_rect(i1,rects1)
        cv2.imwrite(folder+"5_1x_height.png",i1)

    # Abrir morfologicamente
    IF = bin_img.copy()
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(math.ceil(x_h/4),math.ceil(col/4)))
    mask1 = cv2.morphologyEx(IF,cv2.MORPH_OPEN, kernel1)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(math.ceil(row/8),math.ceil(x_h/4)))
    mask2 = cv2.morphologyEx(IF,cv2.MORPH_OPEN, kernel2)
    IW = cv2.bitwise_or(mask1,mask2)
    I_inv = cv2.bitwise_not(bin_img.copy())
    rects2 = []
    for rect in rects:
        x = rect[0]
        y = rect[1]
        w = rect[2]
        h = rect[3]
        if (w > 10*h or h>w*10):
            IW[y:y+h,x:x+w] = 1
            I_inv[y:y+h,x:x+w] = 0
        else:
            rects2.append(rect)
    if write:
        cv2.imwrite(folder+"6mascara_IW.png",IW)

    #Proyecciones
    IW_bin = IW//255
    horizontal_projection = projection(IW_bin.copy(),'h',rec_img,0.75)
    vertical_projection = projection(IW_bin.copy(),'v',rec_img,0.75)

    #Fragmentar
    fragments = []

    if (len(horizontal_projection)>len(vertical_projection)):
        frag1 = fragm(horizontal_projection,'h',rec_img,x_h*2)
        if write:
            save_frag(frag1,folder+"7_h_",I_inv)
        frag2 = []
        for fra in frag1:
            proj_v = projection(IW_bin.copy(),'v',fra,0.98)
            if (len(proj_v) > 0):
                frag2 = frag2 + fragm(proj_v,'v',fra,x_h*6)
            else:
                frag2 = frag2 + fra
        fragments = frag2
            
    else:
        frag1 = fragm(vertical_projection,'v',rec_img,x_h*6)
        frag2 = []
        for fra in frag1:
            proj_h = projection(IW_bin.copy(),'h',fra,0.98)
            if (len(proj_h) > 0):
                frag2 = frag2 + fragm(proj_h,'h',fra,x_h*2)
        fragments = frag2
        
    fragments = clean_frag(fragments,I_inv)
    
    if write:
        save_frag(fragments,"8_t_",I_inv)

    #Fase Map
    cont = 1
    LC = np.zeros((row,col))
    LT = np.zeros((row,col))
    LG = np.zeros((row,col))

    for frag in fragments:
        #New Contornos
        x = frag[0]
        y = frag[1]
        w = frag[2]
        h = frag[3]
        subimg_aux = I_inv[y:y+h,x:x+w].copy()
        f_rects = bbox2(subimg_aux.copy(),h,w)
        if (len(f_rects)<1):
            pass
        if write:
            color_subimg = cv2.cvtColor(np.uint8(subimg_aux.copy()),cv2.COLOR_GRAY2RGB)
            print_rect(color_subimg,f_rects)
            cv2.imwrite(folder+"9_1_"+str(cont)+"contornos.png",color_subimg)
        
        #Clasificacion
        LC,LT,LG = classifi(LC,LT,LG,frag,f_rects,x_h,cont)
        cont+=1

    LT = LT - LC
    LT[LT<0] = 0
    LG = LG - LC
    LG[LG<0] = 0
    LG = LG - LT
    LG[LG<0] = 0
    if write:
        cv2.imwrite(folder+"10lt.png",LT)
        cv2.imwrite(folder+"11lc.png",LC)
        cv2.imwrite(folder+"12lg.png",LG)

    #Formato
    lay_col = np.uint8(LC.copy())
    rect_lc = []
    rect_lc1 = bbox2(lay_col,row,col)
    for rect in rect_lc1:
        if rect[3]//2 >= x_h:
            rect_lc.append(rect)
    
    rect_lc = checkbbox(LC,rect_lc)
    #size_rect_lc = 0
    #while (size_rect_lc!=len(rect_lc)):
    #    size_rect_lc = len(rect_lc)
    #    rect_lc = combine_boxes(rect_lc)
    
    
    if write:
        img_lay_col = i.copy()
        print_rect(img_lay_col,rect_lc,'b')
        cv2.imwrite(folder+"13finallc.png",img_lay_col)

    lay_til = np.uint8(LT.copy())
    rect_til1 = bbox2(lay_til,row,col)
    rect_til = []
    for rect in rect_til1:
        if (rect[3]>x_h*2):
            rect_til.append(rect)
    rect_til = checkbbox(LT,rect_til)      
    #size_rect_tl = 0
    #while (size_rect_tl!=len(rect_til)):
    #    size_rect_tl = len(rect_til)
    #    rect_til = combine_boxes(rect_til)
    
    if write:
        img_lay_til = i.copy()
        print_rect(img_lay_til,rect_til,'r')
        cv2.imwrite(folder+"14finallt.png",img_lay_til)
    
    lay_gra = np.uint8(LG.copy())
    rect_gra1 = bbox2(lay_gra,row,col)
    rect_gra = []
    for rect in rect_gra1:
        if (rect[3]>x_h*4):
            rect_gra.append(rect)
    rect_gra = checkbbox(LG,rect_gra)        
    #size_rect_gra = 0
    #while (size_rect_gra!=len(rect_gra)):
        #size_rect_gra = len(rect_gra)
        #rect_gra = combine_boxes(rect_gra)
    if write:
        img_lay_gra = i.copy()
        print_rect(img_lay_gra,rect_gra,'y')
        cv2.imwrite(folder+"15finallt.png",img_lay_gra)
    
    #imagen_final = i.copy()
    imagen_final = cv2.cvtColor(np.uint8(bin_img.copy()),cv2.COLOR_GRAY2RGB)
    print_rect(imagen_final,rect_gra,'y',8)
    print_rect(imagen_final,rect_til,'r',8)
    print_rect(imagen_final,rect_lc,'b',8)
    cv2.imwrite(folder+"Resultado.png",imagen_final)
    
    return


    #color_img = cv2.cvtColor(np.uint8(bin_img.copy()),cv2.COLOR_GRAY2RGB)
