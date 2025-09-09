public class Student {

    private String name;
    private Course course;

    public Student(String name) {
        this.name = name;
    }

    public void setCourse(Course course) {
        this.course = course;
    }

    public String getName() {
        return name;
    }

    public void printInfo() {
        System.out.println("Student name: " + name);
        System.out.println("Course: " + course.getName());
        System.out.println();
    }
    
}
