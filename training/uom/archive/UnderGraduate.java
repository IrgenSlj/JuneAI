public class UnderGraduate extends Student {
    
    private double GPA;

    public UnderGraduate(String name, double GPA) {
        super(name);
        this.GPA = GPA;
    }

    public double getGPA() {
        return GPA;
    }

    public void printInfo() {
        super.printInfo();
        System.out.println("Undergraduate Student");
        System.out.println("Student GPA: " + GPA);

    }


}
