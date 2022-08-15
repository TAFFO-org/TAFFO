#include "i2c.h"


// ***************************************************************************
// *****                        STATIC FUNCTIONS                         *****
// ***************************************************************************

/**
 * Determine if the i2c master is ready to accept new commands or not.
 * \return 1 if ready, 0 otherwise.
 */
static uint8_t __i2c_ready(void) {
    return ((I2C_0->RREG >> 8) & 0x01);
}

/**
 * Start the i2c master and emits a start sequence on the i2c bus.
 * \return nothing.
 */
static void __i2c_start(void) {
    while(!__i2c_ready());
    I2C_0->WREG = I2C_START_CMD;
}

/**
 * Do not close the i2c transaction and be ready to accept new commands.
 * \return nothing.
 */
static void __i2c_restart(void) {
    while(!__i2c_ready());
    I2C_0->WREG = I2C_RESTART_CMD;
}

/**
 * Close an i2c transaction.
 * \return nothing.
 */
static void __i2c_stop(void) {
    while(!__i2c_ready());
    I2C_0->WREG = I2C_STOP_CMD;
}

/**
 * Write a byte of data and return the status. The function
 * concatenates the write command and the data and writes them
 * to core's register when it is ready.
 * \param w_data the byte to write.
 * \return 0 when success, -1 when failure.
 */
static int __i2c_write_byte(const uint8_t data) {
    int ack, acc_data;

    acc_data = data | I2C_WRITE_CMD;
    while(!__i2c_ready());
    I2C_0->WREG = acc_data;
    while(!__i2c_ready());
    ack = (I2C_0->RREG & 0x0200) >> 9;

    if(ack == 0) return 0;
    else         return -1;
}

/**
 * Retrieve a byte of data from the slave device.
 * \param last_op is used to flag whether the read operation
 * is the last one in a transaction.
 * \return the retrieved byte from the i2c slave.
 */
static uint8_t __i2c_read_byte(const uint8_t last_op) {
    int acc_data = last_op | I2C_READ_CMD;
    while(!__i2c_ready());
    I2C_0->WREG = acc_data;
    while(!__i2c_ready());

    return (I2C_0->RREG & 0xFF);
}


// ***************************************************************************
// *****                        PUBLIC FUNCTIONS                         *****
// ***************************************************************************

/**
 * Set the i2c divisor register. Its value is the number of
 * system clock cycles in a quarter period of the i2c scl.
 * \return nothing.
 */
void i2c_set_freq(void) {
    I2C_0->DVSR = SYS_CLK_FREQ_HZ / I2C_FREQ_HZ / 4;
}

/**
 * Perform a read transaction of a given length.
 * \param dev slave device address.
 * \param bytes pointer to an array of retrieved bytes of data.
 * \param byte_num number of bytes to retrieve.
 * \param restart indicates whether a restart (1) or stop (0) condition
 *        has to be generated at the end of the transaction.
 * \return 0 when success, -1 when failure.
 */
int i2c_read_transaction(const uint8_t dev,
                         uint8_t *bytes,
                         uint8_t byte_num,
                         uint8_t restart) {

    uint8_t dev_byte = (dev << 1) | 0x01; // LSB = 1, read

    __i2c_start();
    int ack = __i2c_write_byte(dev_byte);

    for(uint8_t i = 0; i < byte_num-1; i++) {
        *bytes = __i2c_read_byte(0);
        bytes++;
    }
    *bytes = __i2c_read_byte(1);
    if(restart) __i2c_restart();
    else        __i2c_stop();

    return ack;
}

/**
 * Perform a write transaction of a given length.
 * \param dev slave device address.
 * \param bytes pointer to an array of bytes to write.
 * \param byte_num number of bytes to write.
 * \param restart indicates whether a restart (1) or stop (0) condition
 *        has to be generated at the end of the transaction.
 * \return 0 when success, -1 when failure.
 */
int i2c_write_transaction(const uint8_t dev,
                          uint8_t *bytes,
                          uint8_t byte_num,
                          uint8_t restart) {

    int ack, ack1;
    uint8_t dev_byte = (dev << 1); // LSB = 0, write
    __i2c_start();
    ack = __i2c_write_byte(dev_byte);
    for(uint8_t i = 0; i < byte_num; i++) {
        ack1 = __i2c_write_byte(*bytes);
        ack = ack + ack1;
        bytes++;
    }
    if(restart) __i2c_restart();
    else        __i2c_stop();

    return ack;
}
