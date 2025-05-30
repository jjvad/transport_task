from PyQt5 import QtWidgets
from mainw import Ui_MainWindow  # импорт нашего сгенерированного файла
import solve_problem_methods
import sys


class mywindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(mywindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.label_3.setHidden(True)
        self.ui.label_4.setHidden(True)

        self.ui.pushButton.clicked.connect(self.w_h)
        self.ui.pushButton_2.clicked.connect(self.solve_problem)

    def w_h(self):
        global demands
        global proposals
        global height_matrix
        global width_matrix
        demands = self.ui.lineEdit_2.text().split(' ')
        demands = [int(i) for i in demands]
        proposals = self.ui.lineEdit.text().split(' ')
        proposals = [int(i) for i in proposals]
        width_matrix = len(demands)
        height_matrix = len(proposals)
        self.ui.tableWidget.setColumnCount(width_matrix)
        self.ui.tableWidget.setRowCount(height_matrix)
        for i in range(height_matrix):
            self.ui.tableWidget.setRowHeight(i, 58)
        for i in range(width_matrix):
            self.ui.tableWidget.setColumnWidth(i, 58)

    def solve_problem(self):
        self.ui.label_3.setHidden(False)
        self.ui.label_4.setHidden(False)
        global matrix
        matrix = []
        for i in range(height_matrix):
            matrix.append([])
            for j in range(width_matrix):
                matrix[i].append(int(self.ui.tableWidget.item(i, j).text()))
        solved_matrix, total_cost = solve_problem_methods.solve_transport_problem(proposals, demands, matrix)
        for i in range(height_matrix):
            for j in range(width_matrix):
                self.ui.tableWidget.setItem(i, j, QtWidgets.QTableWidgetItem(str(solved_matrix[i][j])))
        self.ui.label_5.setText(str(total_cost))




app = QtWidgets.QApplication([])
application = mywindow()
application.show()

sys.exit(app.exec())