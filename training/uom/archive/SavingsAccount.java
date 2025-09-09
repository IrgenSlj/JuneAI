public class SavingsAccount extends BankAccount {
    
    private double interestRate;

    public SavingsAccount(double initialAmount, double rate) {
        super(initialAmount);
        interestRate = rate;
    }

    public void setInterestRate(double rate) {
        interestRate = rate;
    }

    public void addInterest() {
        double interest = balance*interestRate;
        balance = balance + interest;
    }

    public void printData() {
        System.out.println("Savings Bank Account");
        System.out.println("Balance: " + balance + " rate: " + interestRate);
    }
}
