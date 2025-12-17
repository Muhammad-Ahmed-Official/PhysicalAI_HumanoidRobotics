import React from 'react';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

export default function CodeTabs({children, ...props}) {
  return (
    <Tabs {...props}>
      {children}
    </Tabs>
  );
}

export { TabItem };