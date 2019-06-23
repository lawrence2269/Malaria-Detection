import tkinter
from tkinter.filedialog import askopenfilename
import tkinter.messagebox as msg
import os
import Malaria_Detection_Functions_1 as malaria

#Creating window and setting title
window = tkinter.Tk()
window.configure(background="grey")
window.title("Malaria Detection App")

# Gets the requested values of the height and widht.
windowWidth = window.winfo_reqwidth()
windowHeight = window.winfo_reqheight()
print("Width",windowWidth,"Height",windowHeight)

# Gets both half the screen width/height and window width/height
positionRight = int(window.winfo_screenwidth()/2 - windowWidth/2)
positionDown = int(window.winfo_screenheight()/2 - windowHeight/2)

window.geometry("+{}+{}".format(positionRight, positionDown))
window.geometry("550x130")

file_String_Var = tkinter.StringVar()
file_String_Var_Hide = tkinter.StringVar()
#Defining functions for buttons
def browseFunc():
    global name
    global label_2
    name = askopenfilename(parent=window,initialdir="/Users/lawrence/Documents",
                           filetypes=[("JPEG","*.jpg"),("PNG","*.png"),("All Files","*")],
                           title="Choose File")
    file_String_Var_Hide.set(os.path.basename(name))
    label_2 = tkinter.Label(window,text=os.path.basename(name),font="Verdana 15",foreground="white",background="grey").place(x=100,y=40)
    file_String_Var.set(name)

def submitImage():
    if(len(file_String_Var.get())==0):
        msg.showwarning("Warning","Please select an image!!!!",parent=window)
    else:
        res = malaria.predict_cell(name)
        msg.showinfo("Result",res,parent=window)
        String_val = " ".join(" " for i in range(0,len(file_String_Var_Hide.get())))
        label_2 = tkinter.Label(window,text=String_val,font="Verdana 15",background="grey",foreground="white").place(x=100,y=40)

# now, create some widgets in the frame
label_1 = tkinter.Label(window,text="Image Upload:",font="Verdana 15 bold",background="grey",foreground="white").place(x=160,y=0)
btn1 = tkinter.Button(window, text = "Browse",font="Verdana 15",fg="blue",command=browseFunc,width=10,height=1).place(x=300,y=4)
btn2 = tkinter.Button(window, text = "Predict",font="Verdana 15",fg="green",command=submitImage,width=30,height=2).place(x=130,y=80)
window.mainloop()
