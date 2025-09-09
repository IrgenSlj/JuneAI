import java.util.ArrayList;
import javax.swing.JOptionPane;

public class Read {
    
    public static void readData(ArrayList<Student> students, ArrayList<Course> courses) {

        boolean more = true;
        while (more) {
            String answer = JOptionPane.showInputDialog("Type of student (1:Student, 2:Graduate)");
            int choice = Integer.parseInt(answer);
            String name = JOptionPane.showInputDialog("Type student name");
            String id = JOptionPane.showInputDialog("Type student id");
            String supervisor = null;

            Student student = new Student();

            if (choice == 2) {
                supervisor = JOptionPane.showInputDialog("Type name of the supervisor");
                student = new GradStudent(name, id, supervisor);
            }

            if (choice == 1) {
                student = new Student(name, id);
            }
            
            String courseName = JOptionPane.showInputDialog("Enter course name");
            
            for (Course course: courses) {
                if (courseName.equals(course.getName()))
                    student.addCourse(course);
            }

            String answer2 = JOptionPane.showInputDialog("Add more students? (1:Yes,2:No)");
            int check = Integer.parseInt(answer2);

            if (check == 2) {
                more = false;
            }

            student.printInfo();
        }
        
    }

}
