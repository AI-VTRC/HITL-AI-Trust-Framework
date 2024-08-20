import tkinter as tk
from tkinter import Scale, Button, Label
from PIL import Image, ImageTk
import argparse


class PhotoTrustApp:
    def __init__(self, root, cav_name, other_cav_name, omega_ij, self_detected_objects, other_detected_objects):

        # need to pass the path of the images being compared.

        self.root = root
        self.root.title("Photo Trust Application")

        # Add labels above the images
        self.label_cav1 = Label(root, text=cav_name, font=("Arial", 14))
        self.label_cav1.grid(row=0, column=0, pady=10)

        # Include omega_ij in the label for the other CAV
        trust_info = f"{other_cav_name} - System Trust Score: {omega_ij:.2f}"
        self.label_cav2 = Label(root, text=trust_info, font=("Arial", 14))
        self.label_cav2.grid(row=0, column=1, pady=10)

        # Load and display the first image
        self.image1 = Image.open(self_detected_objects)
        self.image1 = self.image1.resize((600, 600))  # Resize image as needed
        self.photo1 = ImageTk.PhotoImage(self.image1)
        self.label1 = tk.Label(root, image=self.photo1)
        self.label1.grid(row=1, column=0)

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
        temp_filename = 'temp_trust_value.txt'
        try:
            with open(temp_filename, 'w') as f:
                f.write(str(trust_value))
            print(f"Temporary file created at: {temp_filename}")
        except Exception as e:
            print(f"Failed to create temporary file: {e}")
        self.root.quit()  # Quit after setting the value

    def next_connection(self):
        print("Next connection button pressed")
        self.root.quit()  # This quits the tkinter mainloop
        self.root.destroy()  # This destroys the window, ensuring it closes cleanly


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CAV Trust Monitor")
    parser.add_argument("cav_name", type=str, help="Name of CAV Monitoring Incoming Transmission")
    parser.add_argument("other_cav_name", type=str, help="Name of incoming CAV sender")
    parser.add_argument("omega_ij", type=float, help="Current trust score for the other CAV")
    parser.add_argument("self_detected_image", type=str, help="Path to the image for self detected objects")
    parser.add_argument("other_detected_image", type=str, help="Path to the image for other detected objects")

    args = parser.parse_args()

    root = tk.Tk()
    app = PhotoTrustApp(root, args.cav_name, args.other_cav_name, args.omega_ij, args.self_detected_image,
                        args.other_detected_image)
    root.mainloop()
