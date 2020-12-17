f = open("images\_detection.txt")
det = []
detections = []

for line in f.readlines():
    for el in line.split(', '):
        if el[len(el)-1] == '\n':
            el.replace('\n','')
        det.append(float(el))

for i in range(0,int(len(det)/4)):
    detections.append((det[4*i],det[4*i+1],det[4*i+2],det[4*i+3]))


center = detections[0][0]
print("center (x) is ", center)

# x units per degree
k = (center - detections[3][0])/36.9
print("the constant 'pixels per degree' is:",k)

for i in detections:
    angle = (center - i[0]) / k
    print("angle of ", i , "is:", angle)
