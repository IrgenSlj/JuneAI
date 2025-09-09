import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.*;
import java.util.ArrayList;

public class MyFrame extends JFrame {
    
    private JTextField studentNameField, outputField, courseNameField, outputField2;
    private JButton button, button2, button3;
    private JPanel panel;

    private ArrayList<Student> students = new ArrayList<Student>();
    private ArrayList<Course> courses = new ArrayList<Course>();

    public MyFrame() {

        Course C1 = new Course("Java");
        Course C2 = new Course("Math");
        Course C3 = new Course("Databases");

        courses.add(C1);
        courses.add(C2);
        courses.add(C3);

        panel = new JPanel();

        studentNameField = new JTextField("Enter student name");
        courseNameField = new JTextField("Course title");

        outputField = new JTextField(20);
        outputField2 = new JTextField(10);

        button = new JButton("Create Student");
        button2 = new JButton("Print students");


        panel.add(studentNameField);
        panel.add(courseNameField);
        panel.add(button);
        panel.add(outputField);
        panel.add(button2);

        this.setContentPane(panel);

        ButtonListener listener = new ButtonListener();
        button.addActionListener(listener);
        button2.addActionListener(listener);
        //button3.addActionListener(listener);

        this.setVisible(true);
        this.setSize(400, 400);
        this.setTitle("Student configurator");
        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    class ButtonListener implements ActionListener {

        public void actionPerformed(ActionEvent e) {

            if (e.getSource() == button) {
                String studentName = studentNameField.getText();
                String courseName = courseNameField.getText();
                
                Course selectedCourse = null;
                for (Course course:courses) {
                    if (course.getName().equals(courseName)) {
                        selectedCourse = course;
                    }
                }

                Student newStudent = new Student(studentName);
                newStudent.setCourse(selectedCourse);
                students.add(newStudent);
            } 
            else if (e.getSource() == button2) {
                
                for (Student student: students) {
                    //outputField.setText(student.printInfo());
                    student.printInfo();
                }
            }

        }
    }

}
