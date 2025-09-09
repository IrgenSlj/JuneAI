public class RefrigeratedContainer extends Container {
    
    private double power;

    public RefrigeratedContainer(String code, String destination, double power) {
        super(code, destination);
        this.power = power;
    }

    public double getCharge() {
        return 2000*power;
    }
}
