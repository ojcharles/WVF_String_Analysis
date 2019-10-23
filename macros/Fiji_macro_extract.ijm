run("Bio-Formats", "open=@[01]@ color_mode=Grayscale view=Hyperstack stack_order=XYCZT");
run("Z Project...", "projection=[Max Intensity]");
run("Split Channels");
selectImage([channel]);
run("Rotate 90 Degrees Left");
saveAs("Tiff", [02]);
run("Close All");
run("Quit");
