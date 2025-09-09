public class PhdStudent extends Student {

    private String thesis;

    public PhdStudent (String name, String thesis) {
        super(name);
        this.thesis = thesis;
    }

    public void printInfo() {
        super.printInfo();
        System.out.println("PhD Student");
        System.out.println("Student thesis: " + thesis);
    }

    
}
