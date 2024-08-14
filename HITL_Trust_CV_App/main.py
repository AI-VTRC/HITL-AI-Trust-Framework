import tkinter as tk
from tkinter import Scale, Button, Label
from PIL import Image, ImageTk
import os
import argparse


class PhotoTrustApp:
    def __init__(self, root, folder, self_detected_objects, other_detected_objects):

        # need to pass the path of the images being compared.

        self.root = root
        self.root.title("Photo Trust Application")
        root_connection = os.path.join(r'D:\HITL-AI-Trust-Framework\src\assets\data', folder)

        # Add labels above the images
        self.label_cav1 = Label(root, text="CAV1/User1", font=("Arial", 14))
        self.label_cav1.grid(row=0, column=0, pady=10)

        self.label_cav2 = Label(root, text="CAV2", font=("Arial", 14))
        self.label_cav2.grid(row=0, column=1, pady=10)

        image_path1 = os.path.join(root_connection, f"Car{1}", f"frame_{1}.jpg")

        # Load and display the first image
        self.image1 = Image.open(self_detected_objects)
        self.image1 = self.image1.resize((600, 600))  # Resize image as needed
        self.photo1 = ImageTk.PhotoImage(self.image1)
        self.label1 = tk.Label(root, image=self.photo1)
        self.label1.grid(row=1, column=0)

        image_path2 = os.path.join(root_connection, f"Car{2}", f"frame_{1}.jpg")

        # Load and display the second image
        self.image2 = Image.open(other_detected_objects)
        self.image2 = self.image2.resize((600, 600))  # Resize image as needed
        self.photo2 = ImageTk.PhotoImage(self.image2)
        self.label2 = tk.Label(root, image=self.photo2)
        self.label2.grid(row=1, column=1)

        # Create a scale (slider) to set trust value
        self.scale = Scale(root, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, length=300)
        self.scale.grid(row=2, column=1, sticky="w", pady=(10, 0))

        # Button to override trust with padding
        self.override_button = Button(root, text="Override Trust", command=self.override_trust)
        self.override_button.grid(row=2, column=1, sticky="e", padx=(10, 0))

        # Button to navigate to the next connection
        self.next_button = Button(root, text="Next Connection", command=self.next_connection)
        self.next_button.grid(row=2, column=2, padx=(10, 0))  # Adjust padding as needed

    def override_trust(self):
        trust_value = self.scale.get()
        print(f"Trust value set to: {trust_value}")
        # Implement additional logic here if needed

    def next_connection(self):
        # Logic to handle the next connection can be implemented here
        print("Next connection button pressed")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run CAV Trust Monitor")
    parser.add_argument("folder", type=str, help="Folder name containing images")
    args = parser.parse_args()

    # Initialize the application with the specified folder
    root = tk.Tk()
    app = PhotoTrustApp(root, args.folder, args.self_detected_objects, args.other_detected_objects)
    root.mainloop()
