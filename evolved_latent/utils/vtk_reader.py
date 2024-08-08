import vtk


class VTKReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.reader: vtk.vtkDataReader = None
        self.data: vtk.vtkDataObject = None

    def read(self):
        # Determine the file extension and use the appropriate reader
        if self.file_path.endswith(".vtk"):
            # self.reader = vtk.vtkXMLRectilinearGridReader()
            self.reader = vtk.vtkUnstructuredGridReader()
        elif self.file_path.endswith(".vtu"):
            self.reader = vtk.vtkXMLUnstructuredGridReader()
        elif self.file_path.endswith(".vtp"):
            self.reader = vtk.vtkXMLPolyDataReader()
        elif self.file_path.endswith(".vti"):
            self.reader = vtk.vtkXMLImageDataReader()
        elif self.file_path.endswith(".vtr"):
            self.reader = vtk.vtkXMLRectilinearGridReader()
        elif self.file_path.endswith(".vts"):
            self.reader = vtk.vtkXMLStructuredGridReader()
        elif self.file_path.endswith(".vtm") or self.file_path.endswith(".vtmb"):
            self.reader = vtk.vtkXMLMultiBlockDataReader()
        else:
            raise ValueError("Unsupported file extension")

        self.reader.SetFileName(self.file_path)
        self.reader.Update()
        self.data = self.reader.GetOutput()

    def get_points(self):
        if self._check_data_exists():
            return self.data.GetPoints()

    def get_cells(self):
        if self._check_data_exists():
            return self.data.GetCells()

    def get_point_data(self):
        if self._check_data_exists():
            return self.data.GetPointData()

    def get_cell_data(self):
        if self._check_data_exists():
            return self.data.GetCellData()

    def print_summary(self):
        if self.data is None:
            raise ValueError("No data available. Please read a VTK file first.")
        self.data.PrintSelf(None, vtk.vtkIndent())

    def _check_data_exists(self):
        if self.data is None:
            print("No data available. Please read a VTK file first.")
            return False
        return True
