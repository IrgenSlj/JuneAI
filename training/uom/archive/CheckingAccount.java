public class CheckingAccount extends BankAccount {
    
    private int transcactionsCounter;

    public CheckingAccount(double initialAmount) {
        super(initialAmount);
        transcactionsCounter = 0;
    }
    // Method override
    public void deposit(double amount) {
        transcactionsCounter++;
        super.deposit(amount);

        if (transcactionsCounter > 3) {
            deductFees();
        }
    }

    public void deductFees() {
        balance -= 0.5;
        transcactionsCounter = 0;
    }

    public void printData() {
        System.out.println("Checking Bank Account");
        System.out.println("Balance: " + balance + " free limit: 3 transcactions");
    }
}
