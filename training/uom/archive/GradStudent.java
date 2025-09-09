
public class GradStudent extends Student {

    private String supervisor;

    public GradStudent() {
    }

    public GradStudent(String aName, String anId, String aSupervisor) {
        super(aName, anId);
        supervisor = aSupervisor;
    }
    
    public void printInfo() {
        super.printInfo();
        System.out.println("Student supervisor: " + supervisor);
    }
}
