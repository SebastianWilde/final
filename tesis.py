import implementacion
from datetime import datetime
import time
folder = "resultados/"
for num in range(1,48):
    start_time = datetime.now()
    implementacion.alg3(folder,num,False)
    end_time = datetime.now()
    final_time = end_time - start_time
    f = open(folder+str(num)+"/"+"data.txt",'w')
    f.write('{}'.format(final_time))
    f.close()
    print("Termine:",100*num/48,"%")
    #print(final_time)

print("Final")