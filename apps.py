from Tkinter import * 
import classificationmodel as cf
import numpy as np

root = Tk()
root.title("Multilabel Classficiation")
root.geometry('700x300') 
 
def showText():
	txt = [name.get("1.0",'end-1c')]
	np.savetxt('input.txt', txt, fmt='%s')
	hasil = ", ".join(cf.aplikasi('input.txt'))
	if hasil == "" :
		hasil = "Unknown"
	l3.config(text=hasil)

def cleantxt():
	l3.config(text="")
	name.delete(1.0, END)
 
frame1 = Frame(root)
frame1.pack(fill=BOTH,side=TOP)
l = Label(frame1, text="Input Synopsis")
l.grid(column=0,row=0)
name = Text(frame1, height=10, width=70)
name.grid(column=1,row=0)
button = Button(frame1,command=showText,text="Proccess")
button.grid(column=1,row=2,sticky=N)
button2 = Button(frame1,command=cleantxt,text="Clean")
button2.grid(column=1,row=3,sticky=N)
 
# frame2 = Frame(root)
# frame2.pack(fill=BOTH,side=BOTTOM)
l2 = Label(frame1, text = "Kategori : ")
l2.grid(column=0,row=1)
l3 = Label(frame1)
l3.grid(column=1,row=1,sticky=W)
 
root.mainloop()