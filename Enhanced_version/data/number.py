import os
path = "/home/feng/mepro/li/dataset/SISR/DF2K/HR"
filelist = os.listdir(path)
count=801
for file in filelist:
    print(file)
for file in filelist:   
    Olddir=os.path.join(path,file)  
    if os.path.isdir(Olddir):  
        continue
    filename=os.path.splitext(file)[0]   
    filetype=os.path.splitext(file)[1]  
    Newdir=os.path.join(path,str(count).zfill(4)+filetype)  
    os.rename(Olddir,Newdir)

    count+=1

