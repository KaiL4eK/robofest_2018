#include <ch.h>
#include <hal.h>

#if 1
/********************************/
//      ROS SECTION
/********************************/
#include <ros.h>
#include <std_msgs/String.h>
#include <std_msgs/UInt16.h>


void topic_cb( const std_msgs::UInt16& msg )
{
  (void)msg;
  palToggleLine( LINE_LED1 );
}

ros::NodeHandle ros_node;

std_msgs::String str_msg;
ros::Publisher                      topic_pub("string", &str_msg);

ros::Subscriber<std_msgs::UInt16>   topic_sub("receiver_mysuper", &topic_cb);


static THD_WORKING_AREA(waSpinner, 128);
static THD_FUNCTION(Spinner, arg)
{
  (void)arg;
  chRegSetThreadName("Spinner");

  /* ROS setup */
  ros_node.initNode();

  /* ROS publishers */
  ros_node.advertise(topic_pub);

  /* ROS subscribers */
  ros_node.subscribe(topic_sub);

  str_msg.data = "Hello!";

  while (true)
  {
    ros_node.spinOnce();
//    topic_pub.publish(&str_msg);
    chThdSleepMilliseconds( 1 );
  }
}

/********************************/
#endif

static THD_WORKING_AREA(waBlinker, 128);
static THD_FUNCTION(Blinker, arg)
{
  (void)arg;
  chRegSetThreadName("Red Blinker");

  while (true)
  {
    palToggleLine(LINE_LED3);
    chThdSleepMilliseconds( 500 );
  }
}

PWMConfig pwm3conf = {
    .frequency = 1000000,
    .period    = 10000, /* 1/100 s = 10 ms */
    .callback  = NULL,
    .channels  = {
                  {PWM_OUTPUT_DISABLED, NULL},
                  {PWM_OUTPUT_DISABLED, NULL},
                  {PWM_OUTPUT_ACTIVE_HIGH, NULL},
                  {PWM_OUTPUT_DISABLED, NULL}
                  },
    .cr2        = 0,
    .dier       = 0
};

// Delay in sec = tick/freq

SerialConfig sdcfg = {
      .speed = 460800,
      .cr1 = 0,
      .cr2 = USART_CR2_LINEN,
      .cr3 = 0
    };

#include <chprintf.h>

int main(void)
{
  chSysInit();
  halInit();

//  PWMDriver *pwmDriver      = &PWMD3;

//  pwmStart( pwmDriver, &pwm3conf );
//  palSetLineMode( LINE_LED1, PAL_MODE_ALTERNATE(2) );

  sdStart( &SD7, &sdcfg );

  palSetPadMode( GPIOE, 8, PAL_MODE_ALTERNATE(8) );    // TX
  palSetPadMode( GPIOE, 7, PAL_MODE_ALTERNATE(8) );    // RX

  chThdCreateStatic(waBlinker, sizeof(waBlinker), NORMALPRIO, Blinker, NULL);
  chThdCreateStatic(waSpinner, sizeof(waSpinner), NORMALPRIO, Spinner, NULL);

  while (true)
  {
//    str_msg.data = "Hello";
//    topic_pub.publish(&str_msg);



//    palToggleLine( LINE_LED1 );
//    pwmEnableChannel( pwmDriver, 2, 10000 );
    chThdSleepMilliseconds( 500 );
//    pwmEnableChannel( pwmDriver, 2, 5000 );
//    chThdSleepMilliseconds( 500 );
//    pwmEnableChannel( pwmDriver, 2, 2500 );
//    chThdSleepMilliseconds( 500 );
//    pwmEnableChannel( pwmDriver, 2, 1000 );
//    chThdSleepMilliseconds( 500 );
//
//    chnWrite( &SD7, (uint8_t *)"ads\n\r", 5 );

//    int32_t byte = chnGetTimeout( &SD7, TIME_IMMEDIATE );

//    if ( byte != STM_TIMEOUT )
//    chnPutTimeout( &SD7, byte, TIME_IMMEDIATE );
//      chprintf( (BaseSequentialStream*)&SD7, "Read: %d / %c\n\r", byte, byte);

//    if ( byte )
//    {
//      chnPutTimeout( &SD3, byte, TIME_IMMEDIATE );
//      chnWrite( &SD3, (uint8_t *)"ads\n\r", 5 );
//      palToggleLine( LINE_LED3 );
//    }
  }
}
