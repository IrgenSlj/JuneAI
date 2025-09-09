public class Employee {
    
    private int x;
    private Company employer;

    public void setCompany(Company c) {
        employer = c;
    }

    public void setX(int value) {
        x = value;
    }

    public void printX() {
        System.out.println("X is " + x);
    }
}
