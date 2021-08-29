# This class write down the coordinate of each sample

class CoordRecorder():
    def __init__(self, header = ["u", "v"]) -> None:
        self.header = header
        self.coordiantes = []
        return

    def reset(self):
        self.coordiantes.clear()
        return

    def add_record(self, record):
        self.coordiantes.append(record)
        return

    def write(self, output_path):
        pixel_file = output_path + "/coordinate.csv"
        with open(pixel_file, "w") as pixel_f:
            for item in self.header:
                pixel_f.write(item+" ")
            pixel_f.write("\n")
            for coords in self.coordiantes:
                for coord in coords:
                    pixel_f.write(str(coord) + " ")
                pixel_f.write("\n")
        return
