import tkinter as tk
from tkinter import messagebox

# Function to display entered information
def display_info():
    name = name_var.get()
    age = age_var.get()
    email = email_var.get()
    messagebox.showinfo("User Info", f"Name: {name}\nAge: {age}\nEmail: {email}")

# Create main window
window = tk.Tk()
window.title("User Info")

# Input variables
name_var = tk.StringVar()
age_var = tk.StringVar()
email_var = tk.StringVar()

# Name
tk.Label(window, text="Name").pack()
tk.Entry(window, textvariable=name_var).pack()

# Age
tk.Label(window, text="Age").pack()
tk.Entry(window, textvariable=age_var).pack()

# Email
tk.Label(window, text="Email").pack()
tk.Entry(window, textvariable=email_var).pack()

# Submit button
tk.Button(window, text="Submit", command=display_info).pack()

# Run the window
window.mainloop()





import tkinter as tk
from tkinter import messagebox

# Function to show selected hobbies
def show_hobbies():
    hobbies = ""
    if reading_var.get():
        hobbies += "Reading\n"
    if sports_var.get():
        hobbies += "Sports\n"
    if music_var.get():
        hobbies += "Music\n"

    if hobbies:
        messagebox.showinfo("Your Hobbies", hobbies)
    else:
        messagebox.showinfo("Your Hobbies", "You didn't select any hobbies.")

# Create main window
window = tk.Tk()
window.title("Hobby Selector")

# Variables for checkboxes
reading_var = tk.BooleanVar()
sports_var = tk.BooleanVar()
music_var = tk.BooleanVar()

# Create checkboxes
tk.Label(window, text="Choose your hobbies:").pack()
tk.Checkbutton(window, text="Reading", variable=reading_var).pack()
tk.Checkbutton(window, text="Sports", variable=sports_var).pack()
tk.Checkbutton(window, text="Music", variable=music_var).pack()

# Button to show selected hobbies
tk.Button(window, text="Show Hobbies", command=show_hobbies).pack(pady=10)

# Run the GUI
window.mainloop()




import tkinter as tk
from tkinter import messagebox

# Function to show selected gender
def show_gender():
    selected_gender = gender.get()
    messagebox.showinfo("Selected Gender", f"You selected: {selected_gender}")

# Create main window
window = tk.Tk()
window.title("Gender Selector")

# Variable to hold gender
gender = tk.StringVar(value="Male")

# Widgets
tk.Label(window, text="Select your gender:").pack()
tk.Radiobutton(window, text="Male", variable=gender, value="Male").pack()
tk.Radiobutton(window, text="Female", variable=gender, value="Female").pack()
tk.Radiobutton(window, text="Other", variable=gender, value="Other").pack()
tk.Button(window, text="Show", command=show_gender).pack()

# Run the app
window.mainloop()



import tkinter as tk
from tkinter import messagebox

def show_selection():
    selected = [listbox.get(i) for i in listbox.curselection()]
    messagebox.showinfo("Selected", ", ".join(selected))

window = tk.Tk()
langs = ["Python", "Java", "C++", "JavaScript", "Ruby", "Go", "Swift"]

listbox = tk.Listbox(window, selectmode=tk.MULTIPLE)
scrollbar = tk.Scrollbar(window, command=listbox.yview)
listbox.config(yscrollcommand=scrollbar.set)

for lang in langs:
    listbox.insert(tk.END, lang)

listbox.pack(side=tk.LEFT)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

tk.Button(window, text="Show Selected", command=show_selection).pack()

window.mainloop()




import tkinter as tk
from tkinter import messagebox

def show_title():
    selected = title_var.get()
    messagebox.showinfo("Title Selected", f"You selected: {selected}")

window = tk.Tk()
title_var = tk.StringVar(value="Mr.")
titles = ["Mr.", "Ms.", "Dr.", "Prof."]

tk.Label(window, text="Select Title:").pack()
tk.OptionMenu(window, title_var, *titles).pack()
tk.Button(window, text="Submit", command=show_title).pack()

window.mainloop()




import tkinter as tk

def update_label(val):
    age_label.config(text=f"Age: {val}")

window = tk.Tk()
age_slider = tk.Scale(window, from_=0, to=100, orient=tk.HORIZONTAL, command=update_label)
age_slider.pack()

age_label = tk.Label(window, text="Age: 0")
age_label.pack()

window.mainloop()



import tkinter as tk
from tkinter import filedialog

def open_file():
    path = filedialog.askopenfilename()
    if path:
        file_label.config(text=path)

window = tk.Tk()
tk.Button(window, text="Open File", command=open_file).pack()
file_label = tk.Label(window, text="")
file_label.pack()

window.mainloop()






import tkinter as tk

def calculate_total():
    qty = int(spinbox.get())
    total = qty * 10  # Assume ₹10 per item
    result_label.config(text=f"Total: ₹{total}")

window = tk.Tk()
tk.Label(window, text="Quantity:").pack()
spinbox = tk.Spinbox(window, from_=1, to=10)
spinbox.pack()

tk.Button(window, text="Calculate", command=calculate_total).pack()
result_label = tk.Label(window, text="")
result_label.pack()

window.mainloop()
