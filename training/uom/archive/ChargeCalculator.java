import javax.swing.*;
import java.awt.event.*;

public class ChargeCalculator extends JFrame {

    private JButton calculateChargeButton;
    private JPanel panel;
    private Ship ship;

    public ChargeCalculator(Ship ship) {

        this.ship = ship;
        panel = new JPanel();
        calculateChargeButton = new JButton("Calculate charge");

        panel.add(calculateChargeButton);

        this.setContentPane(panel);

        ButtonListener listener = new ButtonListener();
        calculateChargeButton.addActionListener(listener);

        this.setVisible(true);
        this.setSize(400, 400);
        this.setTitle("Charge calculator");
        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }
    
    class ButtonListener implements ActionListener {

        public void actionPerformed(ActionEvent e) {

            System.out.println(ship.getName() + " total charge: " + ship.getTotalCharge() + "Euro");
        }
    }
}
